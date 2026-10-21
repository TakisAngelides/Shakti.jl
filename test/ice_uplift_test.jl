    @testset "Ice uplift (pw <= po, GlaDS-2 Eq. 4 upper branch)" begin

        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        sl = RegularizedCoulombSlidingLaw(0.25)

        base_mask() = begin
            mask = fill(GROUNDED, nx, ny)
            mask[end, :] .= OCEAN
            mask[1, :]   .= LAND
            mask[:, 1]   .= OTHER_BASIN
            mask
        end
        zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
        zs     = zb .+ 500.0
        G      = fill(0.06, nx, ny)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)
        # The face conductances use the water depth at the start of the step (b + b_uplift), so dt must resolve
        # how fast an uplift layer drains: at dt = 3 h the moulin cell's 0.7 m layer (K up 1e8-fold) empties
        # within the step and Picard cycles; at 30 min it converges and the budget closes to round-off.
        dt     = 1800.0

        # a 3 m^3/s moulin into a 1 mm gap (200 m cells): far more water than the sheet can carry at overburden
        function make_sim(; up = NoUplift(), cf = FilledCavities(), gap = "explicit", e_v = 0.0, mask = base_mask(), moulin = 3.0)
            A_visc = fill(5e-25, nx, ny)
            b0     = fill(1e-3, nx, ny)
            ub_x   = fill(1e-6, nx + 1, ny); ub_y = zeros(nx, ny + 1)
            ieb    = zeros(nx, ny); ieb[3, 3] = moulin / (grid.dx * grid.dy)
            p = ModelParameters(e_v = e_v, b_min = 1e-6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ls = CholeskyDirectSolver(grid)
            kw = e_v == 0 ? (; ps = PicardSolver(500, floattype(1e-12), ls, grid)) : (; pps = ParabolicPicardSolver(500, floattype(1e-12), ls, grid))
            return Simulation(grid, state, 1, floattype(dt), p, gap, String[], ConstantMeltInput(), sl; cavity_filling = cf, uplift = up, kw...)
        end
        grounded(s) = Array(s.mask) .== GROUNDED
        h_o(s, p) = Array(s.zb) .+ Array(s.po) ./ (p.rho_w * p.g)

        function budget_residual(sim, W_old)
            s, p = sim.state, sim.p
            qx, qy = Array(s.q_x), Array(s.q_y)
            div = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
            dW = ((Array(s.b) .- Array(s.b_empty) .+ Array(s.b_uplift)) .- W_old) ./ dt
            return dW .+ div .- (Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
        end
        water(s) = Array(s.b) .- Array(s.b_empty) .+ Array(s.b_uplift)

        @testset "legacy: the moulin drives pw above the overburden" begin
            sim = make_sim()
            step!(sim)
            g = grounded(sim.state)
            @test maximum((Array(sim.state.pw) .- Array(sim.state.po))[g]) > 0
            @test sim.up isa NoUplift
            @test sim.hp isa SheetTopPotential && make_sim(; up = IceUplift()).hp isa SheetTopPotential # the default potential
            @test all(iszero, Array(sim.state.b_uplift)) && all(iszero, Array(sim.state.stor_up))
        end

        for cf in (FilledCavities(), UnfilledCavities())
            @testset "IceUplift with $(nameof(typeof(cf))): pw <= po, uplift layer, budget closes" begin
                sim = make_sim(; up = IceUplift(), cf = cf)
                s, p = sim.state, sim.p
                for k in 1:4
                    W_old = water(s)
                    step!(sim)
                    g = grounded(s)
                    @test first(Shakti.picard_status(sim.hs))
                    @test all(isfinite, Array(s.h))
                    @test maximum((Array(s.pw) .- Array(s.po))[g]) <= 1e-9 * maximum(Array(s.po))
                    @test minimum(Array(s.N)[g]) >= -1e-9 * maximum(Array(s.po))
                    up = g .& (Array(s.h) .> h_o(s, p))
                    @test any(up) # the moulin cell really lifts the ice
                    @test all(Array(s.b_uplift)[up] .> 0)
                    @test all(iszero, Array(s.b_uplift)[g .& .!up])
                    res = budget_residual(sim, W_old)
                    scale = maximum(abs, Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
                    @test maximum(abs, res[g]) < 1e-6 * scale
                end
                # head-based layer: b_uplift = h - h_o where lifted (Filled; Unfilled takes it from the budget, equal at convergence)
                g = grounded(s); up = g .& (Array(s.h) .> h_o(s, p))
                @test Array(s.b_uplift)[up] ≈ (Array(s.h) .- h_o(s, p))[up] rtol = 1e-6
            end
        end

        @testset "uplift row is continuous with the filled row at h = h_o" begin
            q = ModelParameters()
            b, mdot, beta, ub, A, lc, zb0, po0 = 1e-3, 0.0, 0.05, 1e-6, 5e-25, 1e-3, 10.0, 4e6
            ho = zb0 + po0 / (q.rho_w * q.g)
            terms(h, N) = Shakti.cavity_head_terms(0, b, q.b_min, q.b_max, mdot, beta, ub, A, N, lc, h, zb0, po0, 0.0, 0.0,
                                                    q.rho_w, q.rho_i, q.g, q.n, q.n_minus_1_exp, 0.0, 1 / dt)
            # row residual at the head h: gd*h - gs (the head-dependent part of the row, without the flux)
            row(h, N) = (t = terms(h, N); t[2] * h - t[1])
            ϵ = 1e-6
            @test row(ho + ϵ, 0.0) ≈ row(ho - ϵ, 0.0) atol = 1e-9 # filled side evaluated at N = 0 (its value at h_o)
        end

        @testset "closed cell: exact uplift head" begin
            mask = base_mask()
            for (i, j) in ((2, 3), (4, 3), (3, 2), (3, 4)) # closed cell (3, 3) holding the moulin
                mask[i, j] = OTHER_BASIN
            end
            sim = make_sim(; up = IceUplift(), mask = mask, moulin = 0.1) # 1e-7 m/s, more than sliding opens (5e-8 m/s)
            s, p = sim.state, sim.p
            W_old = water(s)
            step!(sim)
            @test s.pw[3, 3] ≈ s.po[3, 3]
            @test s.N[3, 3] ≈ 0 atol = 1e-6
            @test s.b_uplift[3, 3] > 0
            res = budget_residual(sim, W_old)
            @test abs(res[3, 3]) < 1e-9 * abs(Array(s.ieb)[3, 3])
        end

        @testset "parabolic scheme (e_v > 0): pw <= po" begin
            # e_v = 1e-4: the englacial storage below the overburden (e_v*p_o/(rho_w g), 5 cm) fills within the first steps
            sim = make_sim(; up = IceUplift(), e_v = 1e-4, gap = "implicit")
            s = sim.state
            for _ in 1:4
                step!(sim)
            end
            g = grounded(s)
            @test first(Shakti.picard_status(sim.hs))
            @test maximum((Array(s.pw) .- Array(s.po))[g]) <= 1e-9 * maximum(Array(s.po))
            @test any(Array(s.b_uplift)[g] .> 0)
        end

    end
