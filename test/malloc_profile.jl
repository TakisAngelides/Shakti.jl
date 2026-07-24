# Per-tstep allocation profile across the solver options (CholeskyDirectSolver,
# CGIterativeSolver x {SparseAssembledLinearSystem, MatrixFreeLinearSystem} x
# {plain Jacobi, ChebyshevPreconditioner}), to find where step! allocates and
# whether it matters. Meant to run interactively on a compute node (salloc),
# not the login node -- see test/run_malloc_profile.sh for the salloc/srun
# invocation.
#
# Usage: julia -t <nthreads> --project test/malloc_profile.jl [nx] [ny]

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
        error("Unknown solver: \"$solver\" (expected \"cholesky\", \"cg_sals\", or \"cg_mfls\")")
    end

    ps = PicardSolver(500, 1e-6, ls, grid)
    dt = 3600.0 # 1h
    sim = Simulation(grid, state, typemax(Int) - 1, floattype(dt), p, "implicit", String[], mi, sl; ps = ps)

    return sim, ls

end

# Runs f() nwarmup times (JIT + any one-off setup), discards those, then
# measures @allocated on nmeasure further calls -- so the reported numbers
# are steady-state per-call allocation, not compilation noise.
function measure_allocated(f, nwarmup::Int, nmeasure::Int)
    for _ in 1:nwarmup
        f()
    end
    allocs = Int[]
    for _ in 1:nmeasure
        push!(allocs, @allocated f())
    end
    return allocs
end

stats_kb(allocs) = (sort(allocs ./ 1024)[1], sort(allocs ./ 1024)[cld(end, 2)], sort(allocs ./ 1024)[end])

function main()

    nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
    ny = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64
    nwarmup, nmeasure = 5, 20

    println("Grid: $nx x $ny, Threads.nthreads() = $(Threads.nthreads())\n")

    configs = [
        ("Cholesky",              "cholesky", nothing),
        ("CG + SALS (Jacobi)",    "cg_sals",  nothing),
        ("CG + SALS (Chebyshev)", "cg_sals",  4),
        ("CG + MFLS (Jacobi)",    "cg_mfls",  nothing),
        ("CG + MFLS (Chebyshev)", "cg_mfls",  4),
    ]

    @printf("%-24s | %-30s | %-30s\n", "Config", "step! (KB: min/med/max)", "solve_linear_system! (KB: min/med/max)")
    println("-"^24 * " | " * "-"^30 * " | " * "-"^30)

    for (label, solver, cheb) in configs

        sim, ls = build_sim(solver, cheb, nx, ny)

        step_allocs = measure_allocated(() -> step!(sim), nwarmup, nmeasure)
        smin, smed, smax = stats_kb(step_allocs)

        solve_allocs = measure_allocated(() -> Shakti.solve_linear_system!(ls, sim.state, sim.grid, sim.p, sim.kfs, sim.mi), nwarmup, nmeasure)
        lmin, lmed, lmax = stats_kb(solve_allocs)

        @printf("%-24s | %8.2f / %8.2f / %8.2f | %8.2f / %8.2f / %8.2f\n", label, smin, smed, smax, lmin, lmed, lmax)

    end

    # Chebyshev-specific breakdown: isolate estimate_eigenvalue_bounds! (called
    # once per solve via update_chebyshev_bounds!, see preconditioner.jl)
    # from the rest of the CG solve, since it's the leading suspect for any
    # extra allocation the Chebyshev rows above show relative to Jacobi.
    println("\nChebyshev-specific: estimate_eigenvalue_bounds alone")
    for (label, solver) in (("CG + SALS", "cg_sals"), ("CG + MFLS", "cg_mfls"))
        sim, ls = build_sim(solver, 4, nx, ny)
        Shakti.update_diag_precond!(ls.precond_diag, ls.lsy) # populate d before estimate_eigenvalue_bounds reads it via ls.precond.op
        allocs = measure_allocated(() -> Shakti.estimate_eigenvalue_bounds(ls.precond.op, ls.lsy.rhs, ls.precond.nsteps_estimate), nwarmup, nmeasure)
        emin, emed, emax = stats_kb(allocs)
        @printf("  %-22s | %8.2f / %8.2f / %8.2f KB\n", label, emin, emed, emax)
    end

end

main()
