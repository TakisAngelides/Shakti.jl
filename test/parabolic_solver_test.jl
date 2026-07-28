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

        @testset "Simulation constructor requires ls (not ps) when e_v != 0" begin
            p_parabolic = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p_parabolic, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            @test_throws ErrorException Simulation(grid, state, 1, floattype(60.0), p_parabolic, "implicit", String[], ConstantMeltInput(), sl)

            ls = CholeskyDirectSolver(grid)
            sim = Simulation(grid, state, 1, floattype(60.0), p_parabolic, "implicit", String[], ConstantMeltInput(), sl; ls = ls)
            @test sim.hs isa ParabolicHeadScheme
        end

        @testset "parabolic_solver! updates h and keeps every derived field finite" begin
            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            h_before = copy(Array(state.h))

            ls = CholeskyDirectSolver(grid)
            shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
            dt = 60.0
            Shakti.parabolic_solver!(ls, state, grid, p, shs, Arithmetic(), sl, dt)

            @test !(Array(state.h) ≈ h_before) # the moulin input should actually move h
            @test all(isfinite, Array(state.h))
            @test all(isfinite, Array(state.N))
            @test all(isfinite, Array(state.Re))
            @test all(isfinite, Array(state.mdot))
            @test all(isfinite, Array(state.K))
        end

        @testset "run! completes under ParabolicHeadScheme" begin
            p = ModelParameters(e_v = 1e-3)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            ls = CholeskyDirectSolver(grid)
            sim = Simulation(grid, state, 5, floattype(60.0), p, "implicit", ["h", "b"], ConstantMeltInput(), sl;
                             ls = ls, which_observer = "Live", tracked_times = 0:5)
            run!(sim)

            @test all(isfinite, Array(sim.state.h))
            @test all(isfinite, Array(sim.state.b))

            # ParabolicHeadScheme has no Picard loop to report on -- see run.jl's picard_status.
            converged, last_iter = Shakti.picard_status(sim.hs)
            @test converged === missing
            @test last_iter === missing
        end

    end
