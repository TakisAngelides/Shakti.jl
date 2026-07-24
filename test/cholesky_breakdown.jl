# Breaks solve_linear_system!(::CholeskyDirectSolver, ...) into its three
# sub-steps (update_SALS! assembly / cholesky! refactorization / ldiv! solve)
# to find which one accounts for the ~1MB/call seen in malloc_profile.jl.

using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using LinearAlgebra
using Printf

nx, ny = 64, 64
grid = Grid(nx, ny, 1e3, 1e3)
p = ModelParameters(e_v = 0.0)
mi = ConstantMeltInput()
sl = RegularizedCoulombSlidingLaw(0.25)

mask = fill(GROUNDED, nx, ny)
mask[end, :] .= OCEAN
mask[1, :]   .= LAND
mask[:, 1]   .= OTHER_BASIN

A_visc = fill(5e-25, nx, ny)
zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
zs     = zb .+ 500.0
b      = fill(0.01, nx, ny)
G      = fill(0.06, nx, ny)
ub_x   = fill(1e-6, nx + 1, ny)
ub_y   = zeros(nx, ny + 1)
ieb    = zeros(nx, ny)
ieb[nx ÷ 2, ny ÷ 2] = 3 / (grid.dx * grid.dy)
taub_x = zeros(nx + 1, ny)
taub_y = zeros(nx, ny + 1)

state = State(grid)
set_initial_conditions!(state, grid, p, mi, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

ls = CholeskyDirectSolver(grid)
kfs = Arithmetic()

# One real solve first so M/rhs/fact are all populated with genuine (not
# placeholder) values before measuring -- otherwise cholesky!/ldiv! would be
# operating on the constructor's placeholder identity matrix.
Shakti.solve_linear_system!(ls, state, grid, p, kfs, mi)

function measure_allocated(f, nwarmup, nmeasure)
    for _ in 1:nwarmup
        f()
    end
    allocs = Int[]
    for _ in 1:nmeasure
        push!(allocs, @allocated f())
    end
    return allocs
end

nwarmup, nmeasure = 5, 20

update_allocs = measure_allocated(() -> Shakti.update_SALS!(ls.sals, state, grid, p, kfs, mi), nwarmup, nmeasure)
chol_allocs   = measure_allocated(() -> cholesky!(ls.fact, Symmetric(ls.sals.M)), nwarmup, nmeasure)
ldiv_allocs   = measure_allocated(() -> ldiv!(ls.h_vec, ls.fact, ls.sals.rhs), nwarmup, nmeasure)

for (label, allocs) in (("update_SALS! (assembly)", update_allocs), ("cholesky! (refactorize)", chol_allocs), ("ldiv! (solve)", ldiv_allocs))
    kb = sort(allocs ./ 1024)
    @printf("%-28s | %10.2f / %10.2f / %10.2f KB (min/med/max)\n", label, kb[1], kb[cld(end,2)], kb[end])
end
