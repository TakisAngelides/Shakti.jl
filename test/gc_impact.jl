# Does the allocation malloc_profile.jl found actually cost wall time (GC
# overhead), or is it small enough next to the real solve work to not matter?
# Runs many step! calls per config and reports @timed's gctime fraction.

using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using Printf

function build_sim(solver::String, chebyshev_degree, nx::Int, ny::Int)
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

    ls = if solver == "cholesky"
        CholeskyDirectSolver(grid)
    elseif solver == "cg_sals"
        CGIterativeSolver(grid, SparseAssembledLinearSystem; chebyshev_degree)
    elseif solver == "cg_mfls"
        CGIterativeSolver(grid, MatrixFreeLinearSystem; chebyshev_degree)
    else
        error("Unknown solver: \"$solver\"")
    end

    ps = PicardSolver(500, 1e-6, ls, grid)
    dt = 3600.0
    return Simulation(grid, state, typemax(Int) - 1, floattype(dt), p, "implicit", String[], mi, sl; ps = ps)
end

nx, ny = 64, 64
nwarmup, ntimed = 10, 200

configs = [
    ("Cholesky",              "cholesky", nothing),
    ("CG + SALS (Jacobi)",    "cg_sals",  nothing),
    ("CG + SALS (Chebyshev)", "cg_sals",  4),
    ("CG + MFLS (Jacobi)",    "cg_mfls",  nothing),
    ("CG + MFLS (Chebyshev)", "cg_mfls",  4),
]

println("Grid: $nx x $ny, Threads.nthreads() = $(Threads.nthreads()), $ntimed timed tsteps each\n")
@printf("%-24s | %10s | %10s | %8s | %14s\n", "Config", "total (s)", "gc (s)", "gc %", "KB/step")
println("-"^24 * " | " * "-"^10 * " | " * "-"^10 * " | " * "-"^8 * " | " * "-"^14)

for (label, solver, cheb) in configs
    sim = build_sim(solver, cheb, nx, ny)
    for _ in 1:nwarmup
        step!(sim)
    end
    stats = @timed for _ in 1:ntimed
        step!(sim)
    end
    @printf("%-24s | %10.3f | %10.3f | %7.2f%% | %14.1f\n", label, stats.time, stats.gctime, 100 * stats.gctime / stats.time, stats.bytes / ntimed / 1024)
end
