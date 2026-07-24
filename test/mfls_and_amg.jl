# Two diagnostics:
# 1) Why is CG+MatrixFreeLinearSystem(Jacobi) ~10x slower in wall time than
#    CG+SparseAssembledLinearSystem(Jacobi) despite being mathematically the
#    same operator/preconditioner? Compares Picard/CG iteration counts (an
#    algorithmic difference would show up there) against raw matvec cost in
#    isolation (an implementation/overhead difference would show up there),
#    at two grid sizes.
# 2) Would AlgebraicMultigrid.jl (Ruge-Stuben AMG, CPU/SparseMatrixCSC-only,
#    so only usable with SALS not MFLS) beat Jacobi/Chebyshev preconditioning
#    on CG iteration count and wall time, and does that trend improve with
#    grid size the way AMG's mesh-independent convergence would predict?

using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using AlgebraicMultigrid
using Krylov
using LinearAlgebra
using SparseArrays
using Printf

function make_state_and_solvers(nx::Int, ny::Int)
    grid = Grid(nx, ny, 1e3, 1e3)
    p = ModelParameters(e_v = 0.0)
    mi = ConstantMeltInput()
    sl = RegularizedCoulombSlidingLaw(0.25)
    kfs = Arithmetic()

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

    return grid, p, mi, sl, kfs, state
end

function run_to_convergence!(sim, nsteps)
    for _ in 1:nsteps
        step!(sim)
    end
end

println("=" ^ 70)
println("1) MFLS vs SALS: iteration counts and raw matvec cost")
println("=" ^ 70)

for (nx, ny) in ((64, 64), (256, 256))

    println("\n--- Grid $nx x $ny ---")

    grid, p, mi, sl, kfs, state_c = make_state_and_solvers(nx, ny)
    ls_c = CholeskyDirectSolver(grid)
    ps_c = PicardSolver(500, 1e-6, ls_c, grid)
    sim_c = Simulation(grid, state_c, typemax(Int) - 1, floattype(3600.0), p, "implicit", String[], mi, sl; ps = ps_c)
    run_to_convergence!(sim_c, 5)
    @printf("Cholesky:        Picard last_iter = %d\n", ps_c.last_iter)

    grid2, p2, mi2, sl2, kfs2, state_s = make_state_and_solvers(nx, ny)
    ls_s = CGIterativeSolver(grid2, SparseAssembledLinearSystem)
    ps_s = PicardSolver(500, 1e-6, ls_s, grid2)
    sim_s = Simulation(grid2, state_s, typemax(Int) - 1, floattype(3600.0), p2, "implicit", String[], mi2, sl2; ps = ps_s)
    run_to_convergence!(sim_s, 5)
    @printf("CG+SALS(Jacobi): Picard last_iter = %d, CG niter (last solve) = %d\n", ps_s.last_iter, ls_s.ws.stats.niter)

    grid3, p3, mi3, sl3, kfs3, state_m = make_state_and_solvers(nx, ny)
    ls_m = CGIterativeSolver(grid3, MatrixFreeLinearSystem)
    ps_m = PicardSolver(500, 1e-6, ls_m, grid3)
    sim_m = Simulation(grid3, state_m, typemax(Int) - 1, floattype(3600.0), p3, "implicit", String[], mi3, sl3; ps = ps_m)
    run_to_convergence!(sim_m, 5)
    @printf("CG+MFLS(Jacobi): Picard last_iter = %d, CG niter (last solve) = %d\n", ps_m.last_iter, ls_m.ws.stats.niter)

    # Raw matvec cost in isolation, same populated operator, many reps.
    N = nx * ny
    x = rand(N); y = similar(x)
    nreps = 2000

    for _ in 1:10; mul!(y, ls_s.lsy.M, x); end # warmup (JIT)
    t_sals = @elapsed for _ in 1:nreps
        mul!(y, ls_s.lsy.M, x)
    end

    op_m = Shakti.StencilOperator(ls_m.lsy)
    for _ in 1:10; mul!(y, op_m, x); end
    t_mfls = @elapsed for _ in 1:nreps
        mul!(y, op_m, x)
    end

    @printf("Raw matvec (%d reps): SALS (SparseMatrixCSC) = %.4fs (%.2f us/call), MFLS (StencilOperator/@parallel) = %.4fs (%.2f us/call) -- MFLS/SALS ratio = %.1fx\n",
            nreps, t_sals, 1e6 * t_sals / nreps, t_mfls, 1e6 * t_mfls / nreps, t_mfls / t_sals)

end

println()
println("=" ^ 70)
println("2) AMG (Ruge-Stuben) vs Jacobi vs Chebyshev, on CG+SALS")
println("=" ^ 70)

for (nx, ny) in ((64, 64), (256, 256))

    println("\n--- Grid $nx x $ny ---")
    N = nx * ny

    grid, p, mi, sl, kfs, state = make_state_and_solvers(nx, ny)

    # One real solve to populate a genuine (non-placeholder) M/rhs to build
    # the AMG hierarchy and time solves against.
    ls_warm = CGIterativeSolver(grid, SparseAssembledLinearSystem)
    ps_warm = PicardSolver(500, 1e-6, ls_warm, grid)
    sim_warm = Simulation(grid, state, typemax(Int) - 1, floattype(3600.0), p, "implicit", String[], mi, sl; ps = ps_warm)
    run_to_convergence!(sim_warm, 3)

    M, rhs = ls_warm.lsy.M, ls_warm.lsy.rhs
    x0 = vec(copy(state.h))

    # -- Jacobi --
    ws1 = Krylov.CgWorkspace(M, rhs)
    d = copy(ls_warm.precond_diag)
    for _ in 1:3; Krylov.cg!(ws1, M, rhs, copy(x0); M = Diagonal(d), ldiv = true); end
    t_jac = @elapsed Krylov.cg!(ws1, M, rhs, copy(x0); M = Diagonal(d), ldiv = true)
    n_jac = ws1.stats.niter

    # -- Chebyshev (degree 4, rebuilding bounds fresh like a real solve) --
    precond_diag2 = copy(d)
    cheb = Shakti.ChebyshevPreconditioner(M, precond_diag2, 4)
    ws2 = Krylov.CgWorkspace(M, rhs)
    for _ in 1:3
        Shakti.update_chebyshev_bounds!(cheb, rhs)
        Krylov.cg!(ws2, M, rhs, copy(x0); M = cheb, ldiv = true)
    end
    t_cheb = @elapsed begin
        Shakti.update_chebyshev_bounds!(cheb, rhs)
        Krylov.cg!(ws2, M, rhs, copy(x0); M = cheb, ldiv = true)
    end
    n_cheb = ws2.stats.niter

    # -- AMG (Ruge-Stuben), hierarchy rebuilt fresh each solve (M's VALUES
    # change every Picard iteration, same reason Chebyshev's bounds are
    # rebuilt every solve rather than once at construction) --
    ws3 = Krylov.CgWorkspace(M, rhs)
    for _ in 1:3
        ml = ruge_stuben(M)
        P = aspreconditioner(ml)
        Krylov.cg!(ws3, M, rhs, copy(x0); M = P, ldiv = true)
    end
    t_amg = @elapsed begin
        ml = ruge_stuben(M)
        P = aspreconditioner(ml)
        Krylov.cg!(ws3, M, rhs, copy(x0); M = P, ldiv = true)
    end
    n_amg = ws3.stats.niter
    t_amg_setup_only = @elapsed ruge_stuben(M)

    @printf("Jacobi:     niter=%-4d solve=%.5fs\n", n_jac, t_jac)
    @printf("Chebyshev:  niter=%-4d solve=%.5fs\n", n_cheb, t_cheb)
    @printf("AMG:        niter=%-4d solve=%.5fs (of which hierarchy setup alone = %.5fs)\n", n_amg, t_amg, t_amg_setup_only)

end
