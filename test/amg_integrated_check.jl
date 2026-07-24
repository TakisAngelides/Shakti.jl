# Confirms the now-integrated CGIterativeSolver(g, SparseAssembledLinearSystem;
# amg=true) performs the way the standalone AMG prototype predicted, via the
# real run!/step! path (not manual Krylov.cg! calls).

using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using Printf

function build_sim(precond::Symbol, nx::Int, ny::Int)
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

    ls = precond == :jacobi ? CGIterativeSolver(grid, SparseAssembledLinearSystem) :
         precond == :chebyshev ? CGIterativeSolver(grid, SparseAssembledLinearSystem; chebyshev_degree = 4) :
         precond == :amg ? CGIterativeSolver(grid, SparseAssembledLinearSystem; amg = true) :
         error("unknown precond $precond")

    ps = PicardSolver(500, 1e-6, ls, grid)
    return Simulation(grid, state, typemax(Int) - 1, floattype(3600.0), p, "implicit", String[], mi, sl; ps = ps)
end

nx, ny = 256, 256
nwarmup, ntimed = 5, 20

println("Grid $nx x $ny, $ntimed timed tsteps, Threads.nthreads()=$(Threads.nthreads())\n")
@printf("%-12s | %10s | %10s\n", "Precond", "total (s)", "s/step")
for precond in (:jacobi, :chebyshev, :amg)
    sim = build_sim(precond, nx, ny)
    for _ in 1:nwarmup; step!(sim); end
    t = @elapsed for _ in 1:ntimed
        step!(sim)
    end
    @printf("%-12s | %10.3f | %10.4f\n", precond, t, t / ntimed)
end
