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

        function fresh_sim(; p = ModelParameters(e_v = 0.0), gap = "implicit", order = 1, k_face_choice = "arithmetic", alpha = nothing, tsteps = 6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ps = PicardSolver(200, floattype(1e-8), CholeskyDirectSolver(grid), grid; alpha = alpha)
            return Simulation(grid, state, tsteps, floattype(3600.0), p, gap, String[], ConstantMeltInput(), sl;
                              ps = ps, head_extrapolation_order = order, k_face_choice = k_face_choice)
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
            dx, dy = grid.dx, grid.dy
            # Rebuild every face flux from the head the solve produced and the face transmissivities
            # it assembled: q must match face by face -- including the outlet faces into OCEAN/LAND,
            # where it used to come out 8x too small (b averaged with the Dirichlet cell's b = 0).
            for j in 1:ny, i in 2:nx
                @test s.q_x[i, j] ≈ -Kx[i, j] * (h[i, j] - h[i-1, j]) / dx * s.valid_x[i, j] atol = 1e-14
            end
            for j in 2:ny, i in 1:nx
                @test s.q_y[i, j] ≈ -Ky[i, j] * (h[i, j] - h[i, j-1]) / dy * s.valid_y[i, j] atol = 1e-14
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
            @test Shakti.raw_update_max(NoHeadRelaxation(), [0.0, -2e-3, 1e-3, 0.0]) == 2e-3
            @test Shakti.raw_update_max(UnderHeadRelaxation(0.1), [0.0, -2e-3, 1e-3, 0.0]) ≈ 2e-2
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
            # disabled under the parabolic scheme
            p_ev = ModelParameters(e_v = 1e-4)
            st = State(grid)
            set_initial_conditions!(st, grid, p_ev, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            pps = ParabolicPicardSolver(50, floattype(1e-6), CholeskyDirectSolver(grid), grid)
            sim_par = Simulation(grid, st, 2, floattype(3600.0), p_ev, "implicit", String[], ConstantMeltInput(), sl; pps = pps)
            @test sim_par.he isa NoHeadExtrapolation
        end

        @testset "overburden follows the evolving gap height" begin
            sim = fresh_sim(; order = 0)
            step!(sim) # b evolves
            Shakti.prepare_head_solve!(sim) # H/po are refreshed from it before the next head solve
            s = sim.state
            @test Array(s.H) ≈ Array(s.zs) .- (Array(s.zb) .+ Array(s.b)) rtol = 1e-12
            @test Array(s.po) ≈ sim.p.rho_i * sim.p.g .* Array(s.H) rtol = 1e-12
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
            scnc = SmoothCellNClamping(Dict((3, 3) => (0.0, 1.0)), 1e-3)
            apply_cell_N_clamping!(s, scnc)
            @test abs(s.N[3, 3] - 1.0) < 1e-2
            cgc = CellGapClamping(Dict((2, 2) => (0.0, 1e-4)))
            apply_cell_gap_clamping!(s, cgc)
            @test s.b[2, 2] <= 1e-4
        end

        @testset "Chebyshev (Gershgorin bounds) and matrix-free CG agree with Cholesky" begin
            ref = fresh_sim(; order = 0)
            for _ in 1:3; step!(ref); end
            for ls in (CGIterativeSolver(grid, SparseAssembledLinearSystem; amg = false, chebyshev_degree = 4),
                       CGIterativeSolver(grid, MatrixFreeLinearSystem; chebyshev_degree = 4),
                       CGIterativeSolver(grid, MatrixFreeLinearSystem))
                st = State(grid)
                p = ModelParameters(e_v = 0.0)
                set_initial_conditions!(st, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
                sim = Simulation(grid, st, 3, floattype(3600.0), p, "implicit", String[], ConstantMeltInput(), sl;
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

    end
