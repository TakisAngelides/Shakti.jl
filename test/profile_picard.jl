# Single-run profiling script: runs a short simulation and prints the
# TimerOutputs breakdown of where time goes inside step_h!/step_b! (see the
# @timeit PERF_TIMER calls in elliptic_solver.jl, melt_rate.jl, run.jl).
#
# Usage:
#   julia -t <nthreads> --project=. test/profile_picard.jl <solver> <nx> <ny> <tsteps>
#
# <solver> one of: LU, GMRES, GMRES_MF, BiCGSTAB, BiCGSTAB_MF
#
# Example:
#   julia -t 1 --project=. test/profile_picard.jl LU 32 32 10

using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using TimerOutputs

length(ARGS) == 4 || error(
    "Usage: julia -t <nthreads> test/profile_picard.jl <solver> <nx> <ny> <tsteps>\n" *
    "  <solver> one of: LU, GMRES, GMRES_MF, BiCGSTAB, BiCGSTAB_MF"
)

const LINEAR_SOLVER = ARGS[1]
const NX = parse(Int, ARGS[2])
const NY = parse(Int, ARGS[3])
const TSTEPS = parse(Int, ARGS[4])

# Fixed across runs so timing comparisons (solver x thread count x grid size)
# aren't confounded by also varying the physics -- mirrors test/options_explorer.jl's
# baseline_LU config.
const LX, LY = 1e3, 1e3
const DT = 3600.0
const PICARD_ITERS = 500
const PICARD_TOL = 1e-6
const MOULIN_FLUX = 4.0
const MOULIN_RADIUS = 15.0

function build_linear_solver(linear_solver, grid)
    if linear_solver == "LU"
        LUDirectSolver(grid)
    elseif linear_solver == "GMRES"
        GMRESIterativeSolver(grid, SparseAssembledLinearSystem)
    elseif linear_solver == "GMRES_MF"
        GMRESIterativeSolver(grid, MatrixFreeLinearSystem)
    elseif linear_solver == "BiCGSTAB"
        BiCGSTABIterativeSolver(grid, SparseAssembledLinearSystem)
    elseif linear_solver == "BiCGSTAB_MF"
        BiCGSTABIterativeSolver(grid, MatrixFreeLinearSystem)
    else
        error("Unknown solver: $linear_solver. Choose from: LU, GMRES, GMRES_MF, BiCGSTAB, BiCGSTAB_MF")
    end
end

function main()
    println("Profiling: solver=$LINEAR_SOLVER, grid=$(NX)x$(NY), tsteps=$TSTEPS, threads=$(Threads.nthreads())")

    grid  = Grid(NX, NY, LX, LY)
    state = State(grid)
    p     = ModelParameters(e_v = 0.0)
    mi    = ConstantMeltInput()

    # "barrier" mask from options_explorer.jl: downstream obstacle sized/positioned
    # relative to the moulin and grid, so it stays valid across grid sizes.
    im, jm = ceil(Int, NX / 2), ceil(Int, NY / 2)
    mask = fill(GROUNDED, NX, NY)
    offset    = clamp(round(Int, 0.125 * NX), 1, NX - im - 1)
    ic        = im + offset
    halfwidth = clamp(round(Int, NX / 32), 1, min(jm - 2, NY - jm - 1))
    for jc in (jm - halfwidth):(jm + halfwidth)
        mask[ic, jc] = OTHER_BASIN
    end
    mask[1, :]   .= OTHER_BASIN
    mask[end, :] .= OCEAN
    mask[:, 1]   .= OTHER_BASIN
    mask[:, end] .= OTHER_BASIN

    A_visc = fill(5e-25, NX, NY)
    zb     = repeat(reshape(-0.02 .* grid.x, NX, 1), 1, NY)
    zs     = zb .+ 500.0
    b      = fill(0.01, NX, NY)
    G      = fill(0.06, NX, NY)
    ub_x   = fill(1e-6, NX + 1, NY)
    ub_y   = zeros(NX, NY + 1)
    ieb    = zeros(NX, NY)

    xm, ym = grid.x[im], grid.y[jm]
    footprint = [(grid.x[i] - xm)^2 + (grid.y[j] - ym)^2 <= MOULIN_RADIUS^2 for i in 1:NX, j in 1:NY]
    ieb[footprint] .= MOULIN_FLUX / (count(footprint) * grid.dx * grid.dy)

    set_initial_conditions!(state, grid, p, mi, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)

    ls = build_linear_solver(LINEAR_SOLVER, grid)
    ps = PicardSolver(PICARD_ITERS, PICARD_TOL, ls, grid)

    sim = Simulation(grid, state, TSTEPS, floattype(DT), p, "explicit", String[], mi; ps = ps, verbose = true)

    # Warm up (JIT-compile every kernel) before timing. Without this, kernels called
    # once per Picard iteration (~1000s of calls) barely notice the one-time compile
    # cost in their average, but kernels called once per *timestep* (compute_b! and
    # friends in step_b!, only `tsteps` calls total) have it dominate their average.
    println("Warming up (JIT compilation)...")
    for _ in 1:2
        step!(sim)
    end

    reset_timer!(PERF_TIMER)
    @time run!(sim)

    println()
    print_timer(PERF_TIMER; sortby = :time)
    println()
end

main()
