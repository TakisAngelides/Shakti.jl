# Rerun of mfls_and_amg.jl's part 2 with a fixed x0: the first version
# warm-started from state.h, which -- since state.h already IS this exact
# system's converged solution (run_to_convergence! had just solved it) --
# made every preconditioner converge in ~0 iterations and measured nothing.
# Cold start (x0 = 0) instead, plus a 512x512 grid to see the trend AMG's
# mesh-independent convergence predicts more clearly.

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

    return grid, p, mi, sl, state
end

function run_to_convergence!(sim, nsteps)
    for _ in 1:nsteps
        step!(sim)
    end
end

println("=" ^ 70)
println("AMG (Ruge-Stuben) vs Jacobi vs Chebyshev, on CG+SALS -- cold start (x0=0)")
println("=" ^ 70)

for (nx, ny) in ((64, 64), (256, 256), (512, 512))

    println("\n--- Grid $nx x $ny ---")

    grid, p, mi, sl, state = make_state_and_solvers(nx, ny)

    ls_warm = CGIterativeSolver(grid, SparseAssembledLinearSystem)
    ps_warm = PicardSolver(500, 1e-6, ls_warm, grid)
    sim_warm = Simulation(grid, state, typemax(Int) - 1, floattype(3600.0), p, "implicit", String[], mi, sl; ps = ps_warm)
    run_to_convergence!(sim_warm, 3)

    M, rhs = ls_warm.lsy.M, ls_warm.lsy.rhs
    x0 = zeros(eltype(rhs), length(rhs))

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

    # -- AMG (Ruge-Stuben), hierarchy rebuilt fresh each solve since M's
    # VALUES change every Picard iteration (same reason Chebyshev's bounds
    # are rebuilt every solve rather than once at construction) --
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
