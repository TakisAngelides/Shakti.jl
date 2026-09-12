    @testset "ParabolicHeadScheme" begin

        # Same nontrivial mask/state as the linear-solver testsets above.
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
        b      = fill(0.01, nx, ny)
        G      = fill(0.06, nx, ny)
        ub_x   = fill(1e-6, nx + 1, ny)
        ub_y   = zeros(nx, ny + 1)
        ieb    = zeros(nx, ny)
        ieb[3, 3] = 3 / (grid.dx * grid.dy)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)

        @testset "Simulation constructor requires pps (not ps) when e_v != 0" begin
            p_parabolic = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p_parabolic, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            @test_throws ErrorException Simulation(grid, state, 1, floattype(60.0), p_parabolic, "fully_implicit", String[], ConstantMeltInput(), sl)

            ls = CholeskyDirectSolver(grid)
            pps = ParabolicPicardSolver(50, 1e-6, ls, grid)
            sim = Simulation(grid, state, 1, floattype(60.0), p_parabolic, "fully_implicit", String[], ConstantMeltInput(), sl; pps = pps)
            @test sim.hs isa ParabolicHeadScheme
        end

        @testset "parabolic_solver! updates h and keeps every derived field finite" begin
            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            h_before = copy(Array(state.h))

            ls = CholeskyDirectSolver(grid)
            mt = MeltTerms{p.mdot_includes_G, p.mdot_includes_frictional, p.mdot_includes_potential, p.mdot_includes_sensible, p.mdot_includes_qT}()
            dt = 60.0
            Shakti.parabolic_solver!(ls, state, grid, p, mt, Arithmetic(), sl, dt)

            @test !(Array(state.h) ≈ h_before) # the moulin input should actually move h
            @test all(isfinite, Array(state.h))
            @test all(isfinite, Array(state.N))
            @test all(isfinite, Array(state.Re))
            @test all(isfinite, Array(state.mdot))
            @test all(isfinite, Array(state.K))
        end

        @testset "Steady state matches EllipticHeadScheme's (p.e_v -> 0 limit)" begin

            # e_v only sets how fast h relaxes toward the steady balance of diffusion vs.
            # sources (Sommers et al. 2018 Eq. 13's e_v*dh/dt storage term vanishes once
            # dh/dt -> 0) -- it doesn't change what that steady state *is*. Run both schemes
            # from the same initial condition under the same constant forcing, long enough to
            # plateau, and check they land on the same (h, b).
            p_elliptic  = ModelParameters(e_v = 0.0,  b_min = 1e-3)
            p_parabolic = ModelParameters(e_v = 1e-3, b_min = 1e-3)

            state_elliptic  = State(grid)
            state_parabolic = State(grid)
            set_initial_conditions!(state_elliptic,  grid, p_elliptic,  sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            set_initial_conditions!(state_parabolic, grid, p_parabolic, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            dt, tsteps = 3600.0, 1000 # backward-Euler is unconditionally stable, so dt can be large; chosen (see git history) so both schemes fully plateau well within tsteps

            ls_elliptic = CholeskyDirectSolver(grid)
            ps_elliptic = PicardSolver(500, 1e-6, ls_elliptic, grid)
            sim_elliptic = Simulation(grid, state_elliptic, tsteps, floattype(dt), p_elliptic, "fully_implicit", String[], ConstantMeltInput(), sl; ps = ps_elliptic)
            run!(sim_elliptic)

            ls_parabolic = CholeskyDirectSolver(grid)
            pps_parabolic = ParabolicPicardSolver(50, 1e-6, ls_parabolic, grid)
            sim_parabolic = Simulation(grid, state_parabolic, tsteps, floattype(dt), p_parabolic, "fully_implicit", String[], ConstantMeltInput(), sl; pps = pps_parabolic)
            run!(sim_parabolic)

            # Empirically ~3e-7 relative -- rtol here leaves ample margin rather than pinning the exact residual.
            @test Array(state_elliptic.h) ≈ Array(state_parabolic.h) rtol = 1e-4
            @test Array(state_elliptic.b) ≈ Array(state_parabolic.b) rtol = 1e-4

        end

        @testset "run! completes under ParabolicHeadScheme" begin
            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            ls = CholeskyDirectSolver(grid)
            pps = ParabolicPicardSolver(50, 1e-6, ls, grid)
            sim = Simulation(grid, state, 5, floattype(60.0), p, "fully_implicit", ["h", "b"], ConstantMeltInput(), sl;
                             pps = pps, which_observer = "Live", tracked_times = 0:5)
            run!(sim)

            @test all(isfinite, Array(sim.state.h))
            @test all(isfinite, Array(sim.state.b))

            # ParabolicHeadScheme now iterates to nonlinear convergence within each timestep too
            # (ParabolicPicardSolver), so this reports real convergence info, not missing/missing.
            converged, last_iter = Shakti.picard_status(sim.hs)
            @test converged isa Bool
            @test converged # trivial 6x6 fixture, well within 50 iterations
            @test last_iter isa Int
        end

        @testset "run! with a FROZEN_BED cell present keeps it pinned" begin
            # Closes the one gap in the tests above: none of them exercise FROZEN_BED, the
            # biggest architectural addition since this file was last touched. Mirrors
            # frozen_bed_test.jl's own elliptic-path check, but under ParabolicHeadScheme
            # (update_SALS_parabolic_kernel!/update_MFLS_parabolic_kernel! -- see
            # linear_solver.jl -- already branch on FROZEN_BED the same way the elliptic
            # kernels do; this confirms that branch is actually exercised end-to-end).
            mask_frozen = copy(mask) # copy, not mutate: `mask` is shared with the other testsets above
            mask_frozen[3, 3] = FROZEN_BED

            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask_frozen, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            h_before = state.h[3, 3]

            ls = CholeskyDirectSolver(grid)
            pps = ParabolicPicardSolver(50, 1e-6, ls, grid)
            sim = Simulation(grid, state, 5, floattype(60.0), p, "fully_implicit", String[], ConstantMeltInput(), sl; pps = pps)
            run!(sim)

            @test all(isfinite, Array(sim.state.h))
            @test all(isfinite, Array(sim.state.N))
            @test sim.state.h[3, 3] == h_before # frozen row: h held exactly fixed through every backward-Euler solve
            @test sim.state.b[3, 3] == 0.0      # step_b! never touches a non-GROUNDED cell
            @test sim.state.N[3, 3] ≈ sim.state.po[3, 3] # still reads as full overburden throughout
        end

    end
