# Diagnoses the "linear operator or preconditioner is not SPD" crash seen at
# 512x512 with ChebyshevPreconditioner(degree=4, default nsteps_estimate=15).
# Prints the estimated [lambda_min, lambda_max] at a few nsteps_estimate
# values to see whether the default is simply too few Lanczos steps to
# reliably bound the (more ill-conditioned, at this grid size) spectrum.

using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using Krylov
using LinearAlgebra
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

nx, ny = 512, 512

grid, p, mi, sl, state = make_state_and_solvers(nx, ny)
ls = CGIterativeSolver(grid, SparseAssembledLinearSystem)
ps = PicardSolver(500, 1e-6, ls, grid)
sim = Simulation(grid, state, typemax(Int) - 1, floattype(3600.0), p, "implicit", String[], mi, sl; ps = ps)
for _ in 1:3
    step!(sim)
end

M, rhs = ls.lsy.M, ls.lsy.rhs
d = copy(ls.precond_diag)

# NOTE: 512^2 = 262144 unknowns -- deliberately NOT computing a dense eigvals
# reference here (Matrix(DA) alone would be ~550GB). Lanczos-only comparison
# below instead.

for nsteps in (5, 10, 15, 20, 30, 50, 80)
    op = Shakti.JacobiScaledOperator(M, d)
    try
        lmin, lmax = Shakti.estimate_eigenvalue_bounds(op, rhs, nsteps)
        @printf("nsteps_estimate=%-4d lambda_min=%.6e lambda_max=%.6e cond=%.3e\n", nsteps, lmin, lmax, lmax/lmin)
    catch e
        @printf("nsteps_estimate=%-4d FAILED: %s\n", nsteps, sprint(showerror, e))
    end
end

println()
println("Now try full CG solve with ChebyshevPreconditioner at each nsteps_estimate:")
for nsteps in (5, 10, 15, 20, 30, 50, 80)
    precond_diag = copy(d)
    cheb = try
        Shakti.ChebyshevPreconditioner(M, precond_diag, 4; nsteps_estimate = nsteps)
    catch e
        @printf("nsteps_estimate=%-4d constructor FAILED: %s\n", nsteps, sprint(showerror, e))
        continue
    end
    ws = Krylov.CgWorkspace(M, rhs)
    x0 = zeros(eltype(rhs), length(rhs))
    try
        Shakti.update_chebyshev_bounds!(cheb, rhs)
        Krylov.cg!(ws, M, rhs, copy(x0); M = cheb, ldiv = true)
        @printf("nsteps_estimate=%-4d OK, niter=%d, solved=%s\n", nsteps, ws.stats.niter, ws.stats.solved)
    catch e
        @printf("nsteps_estimate=%-4d SOLVE FAILED: %s\n", nsteps, sprint(showerror, e))
    end
end
