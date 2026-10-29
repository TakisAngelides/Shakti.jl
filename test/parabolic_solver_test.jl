    @testset "Parabolic head equation (englacial storage, e_v > 0)" begin

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

        @testset "one PicardSolver for both head equations" begin
            p_parabolic = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p_parabolic, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            @test_throws ErrorException Simulation(grid, state, 1, floattype(60.0), p_parabolic, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift())

            ps = PicardSolver(50, 1e-6, CholeskyDirectSolver(grid), grid)
            sim = Simulation(grid, state, 1, floattype(60.0), p_parabolic, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps)
            @test sim.hs isa PicardHeadScheme && sim.he isa HeadExtrapolation
            @test_throws ErrorException Simulation(grid, state, 1, floattype(60.0), ModelParameters(e_v = -1e-3), ConstantMeltInput(), sl;
                                                   cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps)
        end

        @testset "the Picard loop with englacial storage updates h and keeps every derived field finite" begin
            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            h_before = copy(Array(state.h))

            ps = PicardSolver(50, 1e-9, CholeskyDirectSolver(grid), grid)
            mt = MeltTerms{p.mdot_includes_G, p.mdot_includes_frictional, p.mdot_includes_potential, p.mdot_includes_sensible, p.mdot_includes_qT}()
            @test_throws ErrorException elliptic_solver!(ps, state, grid, p, mt, Arithmetic(), sl) # the storage needs its time step
            elliptic_solver!(ps, state, grid, p, mt, Arithmetic(), sl; dt = 60.0)

            @test ps.converged
            @test Array(ps.h_old) == h_before # the head at the start of the step, taken by the loop
            @test !(Array(state.h) ≈ h_before) # the moulin input should actually move h
            @test all(isfinite, Array(state.h))
            @test all(isfinite, Array(state.N))
            @test all(isfinite, Array(state.Re))
            @test all(isfinite, Array(state.mdot))
            @test all(isfinite, Array(state.K))
        end

        @testset "Steady state matches the elliptic one (p.e_v -> 0 limit)" begin

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
            sim_elliptic = Simulation(grid, state_elliptic, tsteps, floattype(dt), p_elliptic, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps_elliptic)
            run!(sim_elliptic)

            ls_parabolic = CholeskyDirectSolver(grid)
            ps_parabolic = PicardSolver(50, 1e-6, ls_parabolic, grid)
            sim_parabolic = Simulation(grid, state_parabolic, tsteps, floattype(dt), p_parabolic, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps_parabolic)
            run!(sim_parabolic)

            # Empirically ~3e-7 relative -- rtol here leaves ample margin rather than pinning the exact residual.
            @test Array(state_elliptic.h) ≈ Array(state_parabolic.h) rtol = 1e-4
            @test Array(state_elliptic.b) ≈ Array(state_parabolic.b) rtol = 1e-4

        end

        @testset "run! completes with englacial storage" begin
            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            ls = CholeskyDirectSolver(grid)
            ps = PicardSolver(50, 1e-6, ls, grid)
            sim = Simulation(grid, state, 5, floattype(60.0), p, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(),
                             ps = ps, observer = LiveObserver(["h", "b"], 0:5))
            run!(sim)

            @test all(isfinite, Array(sim.state.h))
            @test all(isfinite, Array(sim.state.b))

            converged, last_iter = Shakti.picard_status(sim.hs)
            @test converged isa Bool
            @test converged # trivial 6x6 fixture, well within 50 iterations
            @test last_iter isa Int
        end

        @testset "run! with a FROZEN_BED cell present keeps it pinned" begin
            # Closes the one gap in the tests above: none of them exercise FROZEN_BED, the
            # biggest architectural addition since this file was last touched. Mirrors
            # frozen_bed_test.jl's own elliptic-path check, but with englacial storage
            # (the storage terms are added to GROUNDED rows only; this confirms the frozen
            # row is untouched end-to-end).
            mask_frozen = copy(mask) # copy, not mutate: `mask` is shared with the other testsets above
            mask_frozen[3, 3] = FROZEN_BED

            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask_frozen, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            h_before = state.h[3, 3]

            ls = CholeskyDirectSolver(grid)
            ps = PicardSolver(50, 1e-6, ls, grid)
            sim = Simulation(grid, state, 5, floattype(60.0), p, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps)
            run!(sim)

            @test all(isfinite, Array(sim.state.h))
            @test all(isfinite, Array(sim.state.N))
            @test sim.state.h[3, 3] == h_before # frozen row: h held exactly fixed through every solve
            @test sim.state.b[3, 3] == 0.0      # step_b! never touches a non-GROUNDED cell
            @test sim.state.N[3, 3] ≈ sim.state.po[3, 3] # still reads as full overburden throughout
        end

        @testset "GlaDS-2 sheet: the budget holds the sheet and englacial water, and the water test applies" begin
            # unfilled cavities, uplift, implicit upwind and the consistent coupling (the defaults): W is budgeted less the
            # change of the englacial water e_v*clamp(h - zb, 0, h_o - zb), and the head rows store W
            p = ModelParameters(e_v = 1e-3, b_min = 1e-6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, fill(1e-3, nx, ny), G, ub_x, ub_y, ieb, taub_x, taub_y)
            dt = 3600.0
            sim = Simulation(grid, state, 1, floattype(dt), p, ConstantMeltInput(), sl; ps = PicardSolver(500, floattype(1e-10), LUDirectSolver(grid), grid))
            @test Shakti.water_check_dt(sim) == dt
            s = sim.state
            g = Array(s.mask) .== GROUNDED
            E(h) = p.e_v .* clamp.(h .- Array(s.zb), 0, Array(s.po) ./ (p.rho_w * p.g))
            for _ in 1:3
                W_old, be_old = copy(Array(s.W)), copy(Array(s.b_e)) # b_e: the englacial water the last step ended with
                step!(sim)
                @test first(Shakti.picard_status(sim.hs))
                qx, qy = Array(s.q_x), Array(s.q_y)
                dv = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
                src = Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)
                res = (Array(s.W) .- W_old .+ E(Array(s.h)) .- be_old) ./ dt .+ dv .- src
                @test maximum(abs, res[g]) < 1e-9 * maximum(abs, src[g])
                @test any(E(Array(s.h))[g] .> 0)
                @test Array(s.b_e) == E(Array(s.h)) # stored for the next step
            end
            N = Array(s.N)[g]
            @test all(0 .<= N .<= Array(s.po)[g] .* (1 + 1e-9))

            # The overburden falls between steps (the ice thins by 300 m, as a coupled ice model may make it): the cells
            # whose head is now above the new overburden head hold less englacial water, e_v*p_o/(rho_w g), and the rest
            # drains to the sheet. The step starts from the englacial water the last one ended with, so the budget closes;
            # the englacial water recomputed from the start-of-step head at the new p_o would lose that drainage.
            s.zs .-= 300.0
            W_old, h_old, be_old = copy(Array(s.W)), copy(Array(s.h)), copy(Array(s.b_e))
            step!(sim)
            @test first(Shakti.picard_status(sim.hs))
            @test maximum((be_old .- E(h_old))[g]) > 0.01 # the overburden fell: e_v*dp_o/(rho_w g) of englacial water drains
            qx, qy = Array(s.q_x), Array(s.q_y)
            dv = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
            src = Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)
            res = (Array(s.W) .- W_old .+ E(Array(s.h)) .- be_old) ./ dt .+ dv .- src
            @test maximum(abs, res[g]) < 1e-9 * maximum(abs, src[g])
            @test abs(water_created(sim).created) <= 1e-12 * water_created(sim).input
        end

        @testset "SUHMO's coupled gap (WithDiffusion) with the consistent coupling" begin
            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            sim = Simulation(grid, state, 3, floattype(3600.0), p, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(),
                             ps = PicardSolver(200, floattype(1e-9), CholeskyDirectSolver(grid), grid), diffusion_scheme = WithDiffusion(CholeskyDirectSolver(grid)))
            @test sim.gcp isa ConsistentGapCoupling
            for _ in 1:3
                step!(sim)
                @test first(Shakti.picard_status(sim.hs))
            end
            @test all(isfinite, Array(sim.state.h)) && all(isfinite, Array(sim.state.b))
        end

    end
