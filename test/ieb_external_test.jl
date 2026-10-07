    @testset "ieb = ieb_own + ieb_external (coupled englacial input)" begin

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
        b0     = fill(0.01, nx, ny)
        G      = fill(0.06, nx, ny)
        ub_x   = fill(1e-6, nx + 1, ny)
        ub_y   = zeros(nx, ny + 1)
        ieb    = zeros(nx, ny)
        ieb[3, 3] = 3 / (grid.dx * grid.dy)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)
        ext    = zeros(nx, ny)
        ext[4, 4] = 1 / (grid.dx * grid.dy)

        function fresh_sim(ieb_seed; mi = ConstantMeltInput(), p = ModelParameters(e_v = 0.0))
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb_seed, taub_x, taub_y)
            ps = PicardSolver(200, floattype(1e-8), CholeskyDirectSolver(grid), grid)
            return Simulation(grid, state, 3, floattype(3600.0), p, "implicit", String[], mi, sl; ps = ps)
        end

        @testset "standalone: ieb_external stays zero, ieb is Shakti's own input" begin
            sim = fresh_sim(ieb)
            s = sim.state
            @test all(iszero, Array(s.ieb_external))
            @test Array(s.ieb) == ieb
            @test Array(s.ieb_own) == ieb
            step!(sim)
            @test Array(s.ieb) == ieb
        end

        @testset "set_ieb_external! adds to the own input and survives a step" begin
            sim = fresh_sim(ieb)
            s = sim.state
            set_ieb_external!(s, ext)
            @test Array(s.ieb) == ieb .+ ext
            @test Array(s.ieb_own) == ieb
            step!(sim)
            @test Array(s.ieb) == ieb .+ ext
            set_ieb_external!(s, zeros(nx, ny)) # replaced, not accumulated
            @test Array(s.ieb) == ieb
        end

        @testset "time-varying own input keeps the external part" begin
            mi = SeasonalMeltInput()
            sim = fresh_sim(zeros(nx, ny); mi = mi)
            s = sim.state
            set_ieb_external!(s, ext)
            t = 0.55 * mi.seconds_per_year # inside the melt season
            update_ieb!(mi, s, t)
            own = Array(s.ieb_own)
            @test all(>(0), own)
            @test Array(s.ieb) == own .+ ext
        end

        @testset "external input gives the same solution as the same water seeded as own input" begin
            sim_own = fresh_sim(ieb .+ ext)
            sim_ext = fresh_sim(ieb)
            set_ieb_external!(sim_ext.state, ext)
            for _ in 1:3
                step!(sim_own)
                step!(sim_ext)
            end
            @test Array(sim_ext.state.h) == Array(sim_own.state.h)
            @test Array(sim_ext.state.b) == Array(sim_own.state.b)
        end
    end
