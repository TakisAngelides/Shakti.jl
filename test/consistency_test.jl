    @testset "Flux consistency, gap updates, convergence checks, head extrapolation" begin

        # Same fixture as the other test files: GROUNDED interior, OCEAN/LAND/OTHER_BASIN edges,
        # a moulin at (3, 3).
        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        sl = RegularizedCoulombSlidingLaw(0.25)

        mask = fill(GROUNDED, nx, ny)
        mask[end, :] .= OCEAN
        mask[1, :]   .= LAND
        mask[:, 1]   .= OTHER_BASIN

        A_visc = fill(5e-25, nx, ny)
        zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
        zs     = zb .+ 500.0
        b0     = [0.01 + 0.004 * sin(i + 2j) for i in 1:nx, j in 1:ny] # non-uniform, so face averaging matters
        G      = fill(0.06, nx, ny)
        ub_x   = fill(1e-6, nx + 1, ny)
        ub_y   = zeros(nx, ny + 1)
        ieb    = zeros(nx, ny)
        ieb[3, 3] = 3 / (grid.dx * grid.dy)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)

        function fresh_sim(; p = ModelParameters(e_v = 0.0), gap = FullyImplicitGapScheme(), order = 1, k_face_choice = "arithmetic", alpha = nothing, tsteps = 6, gap_coupling = nothing)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ps = PicardSolver(200, floattype(1e-8), CholeskyDirectSolver(grid), grid; alpha = alpha)
            return Simulation(grid, state, tsteps, floattype(3600.0), p, ConstantMeltInput(), sl; gap_scheme = gap, cavity_filling = FilledCavities(), uplift = NoUplift(),
                              ps = ps, head_extrapolation_order = order, k_face_choice = k_face_choice, gap_coupling = gap_coupling)
        end

        # One step's head solve only (no gap update afterwards), so every field is at the solve's
        # converged state.
        solve_head!(sim) = (Shakti.prepare_head_solve!(sim); Shakti.step_h!(sim.hs, sim); sim)

        @testset "q is the flux the linear system conserves (incl. OCEAN/LAND faces), $kf" for kf in ("arithmetic", "harmonic")
            sim = fresh_sim(; k_face_choice = kf)
            step!(sim)
            solve_head!(sim)
            s = sim.state
            h, Kx, Ky = Array(s.h), Array(s.K_x), Array(s.K_y)
            dbx, dby = Array(s.db_x), Array(s.db_y) # gap jumps of the default SheetTopPotential (zero on Dirichlet faces)
            dx, dy = grid.dx, grid.dy
            # Rebuild every face flux from the head the solve produced, the gap jumps and the face transmissivities
            # it assembled: q must match face by face -- including the outlet faces into OCEAN/LAND,
            # where it used to come out 8x too small (b averaged with the Dirichlet cell's b = 0).
            for j in 1:ny, i in 2:nx
                @test s.q_x[i, j] ≈ -Kx[i, j] * (h[i, j] - h[i-1, j] + dbx[i, j]) / dx * s.valid_x[i, j] atol = 1e-14
            end
            for j in 2:ny, i in 1:nx
                @test s.q_y[i, j] ≈ -Ky[i, j] * (h[i, j] - h[i, j-1] + dby[i, j]) / dy * s.valid_y[i, j] atol = 1e-14
            end
            # Outlet face transmissivity is the grounded cell's own (turbulence-corrected) value.
            p = sim.p
            i = nx # face between cell nx-1 (GROUNDED) and nx (OCEAN)
            for j in 2:ny
                K0 = s.b[nx-1, j]^3 * p.g / (12 * p.nu)
                @test Kx[nx, j] ≈ K0 / (1 + p.omega * s.Re_x[nx, j])
            end
            # |q|/nu == Re exactly on every face (the lag-free quadratic).
            @test Array(abs.(s.q_x)) ./ p.nu ≈ Array(s.Re_x) rtol = 1e-10
            # Mass balance at the converged state: net outflow through each GROUNDED cell's faces
            # equals its source terms (the matrix row) -- checked via the linear system itself.
            sals = sim.hs.ps.ls.sals
            Shakti.update_SALS_elliptic!(sals, s, grid, p, sim.kfs)
            r = sals.M * vec(h) - sals.rhs
            @test maximum(abs, r) / maximum(abs, sals.rhs) < 1e-5
        end

        @testset "outlet cells get the whole outlet-face dissipation" begin
            sim = fresh_sim()
            solve_head!(sim)
            s = sim.state
            j = 3
            i = nx - 1 # GROUNDED, east neighbour OCEAN
            q, dh = Array(s.q_x), Array(s.dhdx)
            qy, dhy = Array(s.q_y), Array(s.dhdy)
            p = sim.p
            expected = p.rho_w * p.g * abs(q[i+1, j] * dh[i+1, j] + q[i, j] * dh[i, j] / 2 + qy[i, j+1] * dhy[i, j+1] / 2 + qy[i, j] * dhy[i, j] / 2)
            @test s.Q_diss[i, j] ≈ expected
        end

        @testset "relax_update: backward Euler for R >= 0, exact exponential for R < 0" begin
            dt = 3600.0
            @test Shakti.relax_update(0.01, 1e-7, 1e-5, dt) == (0.01 + dt * 1e-7) / (1 + dt * 1e-5) # unchanged formula
            for R in (-1e-9, -1e-5, -1e-3) # dt*|R| from tiny to huge (backward Euler blows up at dt*|R| = 1)
                b = Shakti.relax_update(0.01, 1e-7, R, dt)
                exact = 0.01 * exp(-R * dt) + 1e-7 * (exp(-R * dt) - 1) / (-R)
                @test isfinite(b) && b > 0
                @test b ≈ exact rtol = 1e-10
            end
            # continuity at R -> 0 from below
            @test Shakti.relax_update(0.01, 1e-7, -1e-14, dt) ≈ Shakti.relax_update(0.01, 1e-7, 0.0, dt) rtol = 1e-8
            # every gap scheme stays finite and positive under deeply negative N with a long step
            for cls in (StandardCreep(), CreepCutoff())
                b = Shakti.implicit_creep_update(cls, 0.01, 1e-8, -1e-4, 86400.0, 0.05)
                @test isfinite(b) && b > 0.01
                b = Shakti.fully_implicit_creep_update(cls, 0.01, 1e-8, 1e-8, -1e-4, 86400.0, 0.1, 0.05)
                @test isfinite(b) && b > 0.01
            end
        end

        @testset "Picard convergence is judged on the unrelaxed update" begin
            @test Shakti.raw_update_max(NoHeadRelaxation(), [0.0, -2e-3, 1e-3, 0.0], 1.0) == 2e-3
            @test Shakti.raw_update_max(UnderHeadRelaxation(0.1), [0.0, -2e-3, 1e-3, 0.0], 1.0) ≈ 2e-2
            # the stall damping ω of Picard_loop! shrinks the applied update, not Anderson's residual g(x_k) - x_k
            @test Shakti.raw_update_max(NoHeadRelaxation(), [0.0, -2e-3, 1e-3, 0.0], 0.25) ≈ 8e-3
            @test Shakti.raw_update_max(UnderHeadRelaxation(0.1), [0.0, -2e-3, 1e-3, 0.0], 0.5) ≈ 4e-2
            ah = AndersonHeadRelaxation(Grid(2, 2, 1.0, 1.0)); copyto!(ah.fk, [0.0, -2e-3, 1e-3, 0.0])
            @test Shakti.raw_update_max(ah, zeros(2, 2), 0.25) == 2e-3
            # end to end: under-relaxation must not stop earlier (at a less accurate head) than no relaxation
            sim_plain = fresh_sim(; order = 0); step!(sim_plain)
            sim_relax = fresh_sim(; order = 0, alpha = 0.3); step!(sim_relax)
            @test first(Shakti.picard_status(sim_relax.hs))
            @test maximum(abs, Array(sim_relax.state.h) .- Array(sim_plain.state.h)) < 1e-4
        end

        @testset "head extrapolation: same answer, fewer iterations, needs converged history" begin
            sims = Dict(o => fresh_sim(; order = o, tsteps = 8) for o in (0, 1, 2))
            iters = Dict(o => Int[] for o in keys(sims))
            for (o, sim) in sims, _ in 1:8
                step!(sim)
                push!(iters[o], Shakti.picard_status(sim.hs)[2])
            end
            for o in (1, 2)
                @test maximum(abs, Array(sims[o].state.h) .- Array(sims[0].state.h)) < 1e-4
                @test iters[o][1:2] == iters[0][1:2] # no extrapolation until two converged heads exist
                @test sum(iters[o]) <= sum(iters[0])
            end
            he = sims[1].he
            @test he isa HeadExtrapolation && he.count == 2
            reset_head_history!(he)
            @test he.count == 0
            # a failed step clears the history
            Shakti.record_head!(he, sims[1].state, 3600.0, true)
            Shakti.record_head!(he, sims[1].state, 3600.0, true)
            @test he.count == 2
            Shakti.record_head!(he, sims[1].state, 3600.0, false)
            @test he.count == 0
            # with englacial storage too: the storage takes the head at the start of the step, kept before the guess
            p_ev = ModelParameters(e_v = 1e-4)
            st = State(grid)
            set_initial_conditions!(st, grid, p_ev, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ps_ev = PicardSolver(50, floattype(1e-6), CholeskyDirectSolver(grid), grid)
            sim_par = Simulation(grid, st, 2, floattype(3600.0), p_ev, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps_ev)
            @test sim_par.he isa HeadExtrapolation
            for _ in 1:3
                step!(sim_par)
            end
            h_start = copy(Array(st.h))
            Shakti.prepare_head_solve!(sim_par)
            @test Array(ps_ev.h_old) == h_start && Array(st.h) != h_start # the guess moved s.h, not the storage's old head
        end

        @testset "overburden follows the evolving gap height" begin
            sim = fresh_sim(; order = 0)
            step!(sim) # b evolves
            Shakti.prepare_head_solve!(sim) # H/po are refreshed from it before the next head solve
            s = sim.state
            b_up = sim.up isa IceUplift ? max.(Array(s.W) .- Array(s.b), 0) : zero(Array(s.b))
            @test Array(s.H) ≈ max.(Array(s.zs) .- (Array(s.zb) .+ Array(s.b) .+ b_up), 0) rtol = 1e-12
            @test Array(s.po) ≈ sim.p.rho_i * sim.p.g .* Array(s.H) rtol = 1e-12
        end

        @testset "overburden: the uplift layer lifts the ice, and a thin margin gives no negative po" begin
            sim = fresh_sim(; order = 0)
            s = sim.state
            zs, zb, b = Array(s.zs), Array(s.zb), Array(s.b)
            W = copy(b)
            W[3, 3] = b[3, 3] + 2.0 # a 2 m uplift layer
            zs[4, 2] = zb[4, 2] + b[4, 2] / 2 # ice thinner than the gap
            copyto!(s.W, W)
            copyto!(s.zs, zs)
            compute_H!(s, IceUplift())
            compute_po!(s, sim.p)
            H = Array(s.H)
            @test H[3, 3] ≈ zs[3, 3] - zb[3, 3] - b[3, 3] - 2.0 rtol = 1e-12
            @test H[4, 2] == 0 && Array(s.po)[4, 2] == 0
            @test all(H .>= 0)
            compute_H!(s, NoUplift()) # without uplift the water above the gap is not a layer under the ice
            @test Array(s.H)[3, 3] ≈ zs[3, 3] - zb[3, 3] - b[3, 3] rtol = 1e-12
        end

        @testset "per-cell clamps act in place on just the listed cells" begin
            sim = fresh_sim(; order = 0)
            s = sim.state
            N_before = Array(s.N)
            cnc = CellNClamping(Dict((3, 3) => (0.0, 1.0), (4, 2) => (2e6, Inf)))
            apply_cell_N_clamping!(s, cnc)
            N_after = Array(s.N)
            @test N_after[3, 3] == clamp(N_before[3, 3], 0.0, 1.0)
            @test N_after[4, 2] == max(N_before[4, 2], 2e6)
            changed = N_after .!= N_before
            changed[3, 3] = changed[4, 2] = false
            @test !any(changed)
        end

        @testset "the simulation's options apply from construction and after a mask change (refresh_state!)" begin
            # set_initial_conditions! and the State methods of the mask helpers know none of the simulation's options; the
            # Simulation refreshes the derived fields at construction, and its own mask-helper methods after the change
            state = State(grid)
            p = ModelParameters(e_v = 0.0)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            @test Array(state.N)[3, 3] > 1.0 # unclamped from the setup
            sim = Simulation(grid, state, 2, floattype(3600.0), p, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(),
                             ps = PicardSolver(200, floattype(1e-8), CholeskyDirectSolver(grid), grid),
                             cell_N_clamping = CellNClamping(Dict((3, 3) => (0.0, 1.0))))
            @test Array(sim.state.N)[3, 3] == 1.0
            fm = falses(size(mask)); fm[2, 2] = mask[2, 2] == GROUNDED
            freeze_cells!(sim, fm)
            @test Array(sim.state.N)[3, 3] == 1.0
        end

        @testset "AdaptiveTimeStep: the percentile follows mask changes, and no short last step" begin
            state = State(grid)
            p = ModelParameters(e_v = 0.0)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ts = AdaptiveTimeStep(grid, state; use_percentile = true, safety_factor = 1e9, dt_min = 1000.0, dt_max = 3000.0, target_time = 6500.0)
            sim = Simulation(grid, state, 10, floattype(3000.0), p, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(),
                             ps = PicardSolver(200, floattype(1e-8), CholeskyDirectSolver(grid), grid), timestep_scheme = ts)
            # the grounded set after freezing cells, not the one at construction
            fm = falses(size(mask)); fm[2:3, 2:4] .= Array(sim.state.mask)[2:3, 2:4] .== GROUNDED
            freeze_cells!(sim, fm)
            Shakti.update_dt!(sim)
            rate, g = Array(ts.rate_field), Array(sim.state.mask) .== GROUNDED
            @test all(rate[.!g] .== -Inf)
            vals = sort(rate[g]); n = length(vals)
            @test Shakti.rate_statistic(ts.rate_field, ts) == vals[clamp(round(Int, ts.percentile * n), 1, n)]
            # dt is dt_max (safety_factor huge); 6500 s used to end 3000, 3000, 500 (below dt_min), now 3000, 2500, 1000
            dts = Float64[]
            while sim.total_time[] < ts.target_time
                step!(sim); push!(dts, sim.dt[]); sim.total_time[] += sim.dt[] # as run! does
            end
            @test dts == [3000.0, 2500.0, 1000.0]
            @test sim.total_time[] == ts.target_time
        end

        @testset "Chebyshev (Gershgorin bounds) and matrix-free CG (Jacobi, multigrid) agree with Cholesky" begin
            ref = fresh_sim(; order = 0)
            for _ in 1:3; step!(ref); end
            for ls in (CGIterativeSolver(grid, SparseAssembledLinearSystem; amg = false, chebyshev_degree = 4),
                       CGIterativeSolver(grid, MatrixFreeLinearSystem; multigrid = false, chebyshev_degree = 4),
                       CGIterativeSolver(grid, MatrixFreeLinearSystem; multigrid = false),
                       CGIterativeSolver(grid, MatrixFreeLinearSystem))
                st = State(grid)
                p = ModelParameters(e_v = 0.0)
                set_initial_conditions!(st, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
                sim = Simulation(grid, st, 3, floattype(3600.0), p, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(),
                                 ps = PicardSolver(200, floattype(1e-8), ls, grid), head_extrapolation_order = 0)
                for _ in 1:3; step!(sim); end
                @test maximum(abs, Array(st.N) .- Array(ref.state.N)) / maximum(abs, Array(ref.state.N)) < 1e-6
            end
            cheb = ChebyshevPreconditioner(spdiagm(0 => ones(4)), ones(4), 4)
            update_chebyshev_bounds!(cheb, ones(4))
            @test cheb.lambda_max == 2.0 && cheb.lambda_min ≈ 2.0 / 30
        end

        @testset "D is never negative" begin
            sim = fresh_sim(; order = 0)
            s = sim.state
            step!(sim)
            compute_D!(s, sim.p, MeltTerms{true, true, false, true, true}(), WithDiffusion(CholeskyDirectSolver(grid))) # sensible term alone is a heat sink here
            @test all(>=(0), Array(s.D_x)) && all(>=(0), Array(s.D_y))
        end

    
        @testset "limit_freeze_on: a cold bed freezes no more water than the gap holds" begin
            # Strong conductive loss into cold ice (q_T >> G): without the limit, mdot stays
            # negative after the gap reaches b_min and the "frozen" water is expelled into the flow.
            sim = fresh_sim(; tsteps = 30, gap_coupling = ExplicitGapCoupling())
            s = sim.state
            s.q_T .= 50.0   # freezes the ~1 cm gap within the 30 one-hour steps
            for _ in 1:30
                step!(sim)
            end
            gr = Array(s.mask) .== GROUNDED
            b, mdot, mmin, N = Array(s.b), Array(s.mdot), Array(s.mdot_min), Array(s.N)
            @test all(isfinite, N) && all(isfinite, b)
            @test all(b[gr] .>= sim.p.b_min * (1 - 1e-12))
            @test all(mdot[gr] .>= mmin[gr] .- 1e-18)
            @test any(isapprox.(mdot[gr], mmin[gr]; rtol = 1e-10))      # the limit binds somewhere
            @test any(isapprox.(b[gr], sim.p.b_min; rtol = 1e-8))       # and those cells sit at the floor
            @test Array(s.W) == Array(s.b)                               # the floor stays full under ExplicitGapCoupling
        end

        @testset "dry floor: ConsistentGapCoupling under FilledCavities stores the same water as the gap update" begin
            # The freezing patch above with ConsistentGapCoupling: the gap reaches b_min and the water below it drains
            # (W < b). With the floor held full of water the head rows were flat there, and Picard failed from step 14.
            sim = fresh_sim(; tsteps = 30, gap_coupling = ConsistentGapCoupling())
            @test Shakti.dry_floor(sim)
            s, p = sim.state, sim.p
            s.q_T .= 50.0
            gr = Array(s.mask) .== GROUNDED
            for _ in 1:30
                W_old = Array(s.W)
                step!(sim)
                @test first(Shakti.picard_status(sim.hs))
                qx, qy = Array(s.q_x), Array(s.q_y)
                div = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
                res = (Array(s.W) .- W_old) ./ sim.dt[] .+ div .- Array(s.mdot) ./ p.rho_w .- Array(s.ieb)
                scale = maximum(abs, Array(s.mdot)[gr] ./ p.rho_w .+ Array(s.ieb)[gr]) + maximum(Array(s.beta)[gr] .* Array(s.abs_ub)[gr])
                @test maximum(abs, res[gr]) < 1e-5 * scale # per-cell water budget
            end
            b, bd, md, mm = Array(s.b), Array(empty_gap(s)), Array(s.mdot), Array(s.mdot_min)
            @test all(b[gr] .>= p.b_min * (1 - 1e-12))
            @test all(0 .<= bd .<= b)
            @test any(bd[gr] .> 0)                                        # some floor really is dry
            @test all(md[gr] .>= mm[gr] .- 1e-18)                         # freezing takes no more than the water present
        end

end
