# Does AMG's CG-iteration-count advantage over Jacobi (clear at a
# near-initial-condition state, see test/amg_rerun.jl: 5-6 iterations vs
# Jacobi's 192-745, essentially flat across grid sizes) hold once the state
# has evolved toward the channelized regime, where hydraulic conductivity K
# becomes strongly heterogeneous (orders of magnitude between channelized
# and distributed regions)? Classical Ruge-Stuben AMG's mesh-independence
# guarantee assumes fairly mild coefficient variation, so this is a real
# open question, not a foregone conclusion.
#
# Physical setup borrowed from test/reproduce_section_3_3.jl (reproduces
# Sommers, Rajaram & Morlighem 2018 GMD Sect. 3.3), at a smaller/faster grid
# (32x32, the paper's own "Cholesky-sized" default) so a long enough
# trajectory to reach channelization onset is cheap to run on CPU. Runs ONE
# reference trajectory (Cholesky -- always robust/correct) and at each
# checkpoint, freezes that exact tstep's assembled (M, rhs) and solves it
# with Jacobi/Chebyshev/AMG directly (cold start, x0=0) -- an apples-to-apples
# comparison on an identical linear system, not three separately-evolving
# trajectories that could drift apart.

using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using Random
using Krylov
using LinearAlgebra
using AlgebraicMultigrid
using Printf

nx, ny = 32, 32
lx, ly = 4000.0, 8000.0
thickness_min, thickness_max = 550.0, 700.0
A_rate = 5e-25
dt = 3600.0
spinup_days = 4.0
run_days = 300.0 # 60 days reached a flat steady state (K_ratio constant at 4.06 throughout) with no channelization -- the paper's own reference snapshot days (test/reproduce_section_3_3.jl) cluster around days 150-280, so the interesting dynamics need most of the seasonal cycle to develop
picard_iters, picard_tol, alpha = 500, 1e-6, 0.1 # alpha=0.1: paper's own fix for Picard convergence once channelization onsets
omega = 0.0001
C = 0.25
seed = 1
checkpoint_every_days = 10.0

grid = Grid(nx, ny, lx, ly)
state = State(grid)
p = ModelParameters(e_v = 0.0, b_min = 1e-3, omega = omega)

mask = fill(GROUNDED, nx, ny)
mask[1, :]   .= LAND
mask[end, :] .= OTHER_BASIN
mask[:, 1]   .= OTHER_BASIN
mask[:, end] .= OTHER_BASIN

zb = zeros(nx, ny)
Hx = sqrt.(thickness_min^2 .+ (thickness_max^2 - thickness_min^2) .* (grid.x ./ lx))
zs = repeat(Hx, 1, ny)

A_visc = fill(A_rate, nx, ny)
b = fill(0.01, nx, ny)
rng = Random.MersenneTwister(seed)
b .*= 1 .+ 0.01 .* randn(rng, nx, ny)
G = fill(0.05, nx, ny)
ub_x = fill(1e-6, nx + 1, ny)
ub_y = zeros(nx, ny + 1)

sl = RegularizedCoulombSlidingLaw(C)
taub_x = zeros(nx + 1, ny)
taub_y = zeros(nx, ny + 1)

seconds_per_year = 365 * 86400.0
mi_spinup = ConstantMeltInput()
ieb_spinup = fill(1.0 / seconds_per_year, nx, ny)

set_initial_conditions!(state, grid, p, mi_spinup, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb_spinup, taub_x, taub_y)

ls = CholeskyDirectSolver(grid)
ps = PicardSolver(picard_iters, picard_tol, ls, grid; alpha = alpha)

spinup_tsteps = round(Int, spinup_days * 86400 / dt)
sim_spinup = Simulation(grid, state, spinup_tsteps, floattype(dt), p, "implicit", String[], mi_spinup, sl; ps = ps)
println("Spin-up: $spinup_tsteps steps")
@elapsed run!(sim_spinup)
println("Spin-up done.\n")

mi_seasonal = SeasonalMeltInput()
initialize_ieb!(mi_seasonal, state, ieb_spinup)

run_tsteps = round(Int, run_days * 86400 / dt)
sim = Simulation(grid, state, run_tsteps, floattype(dt), p, "implicit", String[], mi_seasonal, sl; ps = ps)

checkpoint_every = round(Int, checkpoint_every_days * 86400 / dt)

@printf("%-6s | %8s %8s %8s | %10s %10s %10s | %8s\n", "day", "Jacobi_n", "Cheb_n", "AMG_n", "Jacobi_ms", "Cheb_ms", "AMG_ms", "K_ratio")

for t in 1:run_tsteps

    step!(sim)
    sim.total_time[] += sim.dt # run! normally does this; bypassing run! here for manual per-tstep checkpointing means it has to be done by hand, or update_ieb!(mi_seasonal, ...) below always sees total_time=0 and the seasonal forcing never varies

    if t % checkpoint_every == 0

        M = copy(sim.hs.ps.ls.sals.M)
        rhs = copy(sim.hs.ps.ls.sals.rhs)
        N = nx * ny
        idxP = sim.hs.ps.ls.sals.idxP
        d = zeros(eltype(rhs), N)
        for i in eachindex(idxP)
            d[i] = M.nzval[idxP[i]]
        end
        x0 = zeros(eltype(rhs), N)

        ws1 = Krylov.CgWorkspace(M, rhs)
        t_jac = @elapsed Krylov.cg!(ws1, M, rhs, copy(x0); M = Diagonal(d), ldiv = true)
        n_jac = ws1.stats.niter

        precond_diag2 = copy(d)
        cheb = Shakti.ChebyshevPreconditioner(M, precond_diag2, 4)
        ws2 = Krylov.CgWorkspace(M, rhs)
        Shakti.update_chebyshev_bounds!(cheb, rhs)
        t_cheb = @elapsed Krylov.cg!(ws2, M, rhs, copy(x0); M = cheb, ldiv = true)
        n_cheb = ws2.stats.niter

        ws3 = Krylov.CgWorkspace(M, rhs)
        ml = ruge_stuben(M)
        Pamg = aspreconditioner(ml)
        t_amg = @elapsed Krylov.cg!(ws3, M, rhs, copy(x0); M = Pamg, ldiv = true)
        n_amg = ws3.stats.niter

        K = Array(sim.state.K)
        Kmax, Kmin = maximum(K), minimum(filter(>(0), K))
        k_ratio = Kmax / Kmin

        day = t * dt / 86400
        @printf("%-6.1f | %8d %8d %8d | %10.2f %10.2f %10.2f | %8.2e\n", day, n_jac, n_cheb, n_amg, 1000t_jac, 1000t_cheb, 1000t_amg, k_ratio)

    end

end
