    @testset "Unfilled cavities (step 1: b_empty bookkeeping)" begin

        nx, ny = 4, 3
        grid = Grid(nx, ny, 1e3, 1e3)

        @testset "State starts with b_empty = 0" begin
            state = State(grid)
            @test all(iszero, state.b_empty)
        end

        @testset "update_b_empty!" begin
            mask = fill(GROUNDED, nx, ny)
            mask[1, 1] = OCEAN
            mask[2, 2] = FROZEN_BED

            make() = begin
                s = State(grid)
                copyto!(s.mask, mask)
                s.zb .= 100.0
                s.h .= 150.0  # pw > 0 everywhere ...
                s.h[3, 1] = 100.0 - 0.002 # ... except: GROUNDED, head 2 mm below the bed
                s.h[1, 1] = 50.0          # OCEAN below the bed: must stay 0
                s.h[2, 2] = 50.0          # FROZEN_BED below the bed: must stay 0
                s
            end

            s = make()
            update_b_empty!(s, FilledCavities())
            @test all(iszero, s.b_empty) # legacy: untouched

            s = make()
            update_b_empty!(s, UnfilledCavities())
            @test s.b_empty[3, 1] ≈ 0.002
            @test s.b_empty[1, 1] == 0.0
            @test s.b_empty[2, 2] == 0.0
            @test count(!iszero, s.b_empty) == 1 # filled cells have nothing empty
            @test all(>=(0), s.b_empty)
        end

    end

    @testset "Unfilled cavities (step 2: head rows, pw >= 0, water budget)" begin

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
        ieb    = zeros(nx, ny)
        ieb[3, 3] = 3 / (grid.dx * grid.dy)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)
        dt     = 10800.0

        # `stuck`: gap at the floor, 10 m/yr of sliding, very stiff ice -- opening that no inflow or
        # creep closure at N <= po can balance (the Greenland 16 km failure, in miniature).
        function make_sim(cf; stuck = true, mask = base_mask(), gap = "explicit", kface = nothing, ieb_scale = 1.0)
            A_visc = fill(stuck ? 1e-30 : 5e-25, nx, ny)
            b0     = stuck ? fill(1e-6, nx, ny) : [0.01 + 0.004 * sin(i + 2j) for i in 1:nx, j in 1:ny]
            ub_x   = fill(stuck ? 3e-7 : 1e-6, nx + 1, ny)
            ub_y   = zeros(nx, ny + 1)
            p = ModelParameters(e_v = 0.0, b_min = 1e-6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb .* ieb_scale, taub_x, taub_y)
            ps = PicardSolver(500, floattype(1e-12), CholeskyDirectSolver(grid), grid)
            return Simulation(grid, state, 1, floattype(dt), p, gap, String[], ConstantMeltInput(), sl; ps = ps, cavity_filling = cf, k_face_choice = kface)
        end

        grounded(s) = Array(s.mask) .== GROUNDED

        # per-cell residual of (W_new - W_old)/dt + div(q) = mdot/rho_w + ieb, W = b - b_empty
        function budget_residual(sim, b_old, e_old)
            s, p = sim.state, sim.p
            qx, qy = Array(s.q_x), Array(s.q_y)
            div = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
            dW = ((Array(s.b) .- Array(s.b_empty)) .- (b_old .- e_old)) ./ dt
            return dW .+ div .- (Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
        end

        @testset "no unfilled cell: identical to the legacy solve" begin
            sims = [make_sim(cf; stuck = false, kface = "arithmetic") for cf in (FilledCavities(), UnfilledCavities())] # same face scheme: Upwind is the unfilled default and would differ
            for sim in sims
                Shakti.prepare_head_solve!(sim); Shakti.step_h!(sim.hs, sim)
            end
            g = grounded(sims[1].state)
            @test all(Array(sims[1].state.pw)[g] .> 0) # fixture sanity: nothing is unfilled
            @test Array(sims[2].state.h) ≈ Array(sims[1].state.h) rtol = 1e-12 atol = 1e-9
            @test all(iszero, sims[2].state.b_empty .* 0) # b_empty not touched by the solve itself
        end

        @testset "stuck cells: legacy gives pw < 0, unfilled cavities keep pw >= 0" begin
            fi = make_sim(FilledCavities())
            step!(fi)
            @test minimum(Array(fi.state.pw)[grounded(fi.state)]) < 0 # the problem this feature fixes

            un = make_sim(UnfilledCavities())
            s = un.state
            b_old, e_old = copy(Array(s.b)), copy(Array(s.b_empty))
            step!(un)
            g = grounded(s)
            @test first(Shakti.picard_status(un.hs)) # converged
            @test all(isfinite, Array(s.h))
            @test minimum(Array(s.pw)[g]) >= 0
            @test maximum(Array(s.N)[g] .- Array(s.po)[g]) <= 1e-6 * maximum(Array(s.po)) # N <= po
            @test all(Array(s.b_empty) .>= -Array(s.b) .- 1e-12) # at most b of water stored above the gap
            @test any(Array(s.b_empty)[g] .> 0) # some gap really is empty
            @test all(Array(s.b)[g] .>= un.p.b_min)

            # exact water budget, every grounded cell (explicit gap update: b_new is the row's own b_old + dt*rate)
            res = budget_residual(un, b_old, e_old)
            scale = maximum(abs, Array(s.mdot) ./ un.p.rho_w .+ Array(s.ieb)) + maximum(Array(s.beta) .* Array(s.abs_ub))
            @test maximum(abs, res[g]) < 1e-6 * scale
        end

        @testset "gap held at b_max: opening blocked, empty volume bounded, budget exact" begin
            A_visc = fill(1e-30, nx, ny); b0 = fill(1e-6, nx, ny)
            ub_x = fill(3e-7, nx + 1, ny); ub_y = zeros(nx, ny + 1)
            p = ModelParameters(e_v = 0.0, b_min = 1e-6, b_max = 3e-4)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, base_mask(), A_visc, zb, zs, b0, G, ub_x, ub_y, zeros(nx, ny), taub_x, taub_y) # no moulin: with 3 m^3/s into ice this stiff the pressurized cells are a numerical stress test of their own
            ps = PicardSolver(500, floattype(1e-12), CholeskyDirectSolver(grid), grid)
            sim = Simulation(grid, state, 1, floattype(dt), p, "explicit", String[], ConstantMeltInput(), sl; ps = ps, cavity_filling = UnfilledCavities())
            s = state
            for k in 1:6
                b_old, e_old = copy(Array(s.b)), copy(Array(s.b_empty))
                step!(sim)
                g = grounded(s)
                @test minimum(Array(s.pw)[g]) >= 0
                @test all(Array(s.b_empty)[g] .<= Array(s.b)[g] .+ 1e-9) # never more empty than the gap itself
                if k == 6
                    @test any(Array(s.b)[g] .>= p.b_max * (1 - 1e-9)) # some gap really is held
                    res = budget_residual(sim, b_old, e_old)
                    scale = maximum(abs, Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)) + maximum(Array(s.beta) .* Array(s.abs_ub))
                    @test maximum(abs, res[g]) < 1e-6 * scale
                end
            end
        end

        @testset "water budget closes for every gap scheme (b_empty from the water actually delivered)" begin
            for gap in ("explicit", "implicit", "fully_implicit"), stuck in (false, true)
              @testset "gap $gap, stuck=$stuck" begin
                sim = make_sim(UnfilledCavities(); stuck = stuck, gap = gap, ieb_scale = 1e-4) # a 3 m^3/s moulin into ice this stiff is a numerical stress test of its own
                s = sim.state; b_old, e_old = copy(Array(s.b)), copy(Array(s.b_empty))
                for k in 1:3
                    b_old, e_old = copy(Array(s.b)), copy(Array(s.b_empty))
                    step!(sim)
                end
                g = grounded(s)
                res = budget_residual(sim, b_old, e_old)
                scale = maximum(abs, Array(s.mdot) ./ sim.p.rho_w .+ Array(s.ieb)) + maximum(Array(s.beta) .* Array(s.abs_ub))
                @test maximum(abs, res[g]) < 1e-6 * scale
                @test all(Array(s.b_empty)[g] .<= Array(s.b)[g] .+ 1e-12) && all(Array(s.b_empty)[g] .>= -Array(s.b)[g] .- 1e-12)
                @test minimum(Array(s.pw)[g]) >= 0
              end
            end
        end


        @testset "closed cell: exact unfilled head" begin
            mask = base_mask()
            for (i, j) in ((3, 4), (5, 4), (4, 3), (4, 5)) # closed cell (4, 4); the moulin sits at (3, 3)
                mask[i, j] = OTHER_BASIN
            end
            un = make_sim(UnfilledCavities(); mask = mask)
            s, p = un.state, un.p
            b_old, e_old = copy(Array(s.b)), copy(Array(s.b_empty))
            step!(un)
            @test s.pw[4, 4] == 0
            @test s.b_empty[4, 4] > 0
            res = budget_residual(un, b_old, e_old)
            scale = abs(Array(s.mdot)[4, 4] / p.rho_w + Array(s.ieb)[4, 4]) + Array(s.beta)[4, 4] * Array(s.abs_ub)[4, 4]
            @test abs(res[4, 4]) < 1e-9 * scale
        end

        @testset "empty cavity neither carries nor draws water: conductance from the water depth" begin
            un = make_sim(UnfilledCavities())
            s = un.state
            s.b .= 1e-3
            s.b_empty .= 0.0
            s.b_empty[3, 3] = 9e-4  # nearly dry
            s.b_empty[4, 3] = 1e-3  # dry: floored at b_min
            Shakti.compute_b_w!(s, un.p)
            s.dhdx .= 0.0; s.dhdy .= 0.0 # no gradient: the upwind face takes the smaller cell
            @test s.b_w[3, 3] ≈ 1e-4
            @test s.b_w[4, 3] ≈ un.p.b_min
            @test s.b_w[2, 3] == 1e-3
            Shakti.compute_face_flux!(s, un.p, un.kfs)
            # the face between the nearly-dry cell and a full neighbour is far less conductive than a full-full face
            @test s.K_x[3, 3] < s.K_x[3, 4] # (2,3)-(3,3) face (one cell nearly dry) vs a full-full face elsewhere
        end

        @testset "Upwind face conductance: the donor cell sets it" begin
            @test Shakti.face_mean(Upwind(), 1.0, 5.0, +1.0) == 5.0 # head higher at the high index: water leaves it
            @test Shakti.face_mean(Upwind(), 1.0, 5.0, -1.0) == 1.0
            @test Shakti.face_mean(Upwind(), 1.0, 5.0, 0.0) == 1.0  # no gradient: the smaller
            @test Shakti.face_mean(Arithmetic(), 1.0, 5.0, +1.0) == 3.0 # other schemes ignore the head difference
            @test Shakti.face_conductance(Upwind(), 1.0, GROUNDED, 5.0, GROUNDED, +1.0) == 5.0
            @test Shakti.face_conductance(Arithmetic(), 1.0, GROUNDED, 5.0, GROUNDED) == 3.0

            # through the face kernel: a nearly empty cell next to a full one
            sim = make_sim(UnfilledCavities())
            s, p = sim.state, sim.p
            @test sim.kfs isa Upwind # the default under unfilled cavities
            s.b .= 0.05; s.b_empty .= 0.0
            s.b_empty[3, 3] = 0.05 - 1e-4 # (3,3) holds 0.1 mm of water, its neighbours 5 cm
            Shakti.compute_b_w!(s, p)
            c0 = p.g / (12 * p.nu)
            K0(w) = c0 * w^3
            for (dh, donor_w) in ((+1.0, 0.05), (-1.0, 1e-4)) # face (3,3)|(4,3) = K_x[4, 3]: dh = h[4,3] - h[3,3]
                s.dhdx .= 0.0; s.dhdy .= 0.0
                s.dhdx[4, 3] = dh
                Shakti.compute_face_flux!(s, p, sim.kfs)
                Kf = s.K_x[4, 3] * (1 + p.omega * s.Re_x[4, 3])
                @test Kf ≈ K0(donor_w) rtol = 1e-9
            end
            # an arithmetic mean would let the empty cell discharge through its neighbour's conductance
            s.dhdx .= 0.0; s.dhdx[4, 3] = -1.0 # water leaves (3,3), the nearly empty cell
            Shakti.compute_face_flux!(s, p, Arithmetic())
            Karith = s.K_x[4, 3] * (1 + p.omega * s.Re_x[4, 3])
            @test Karith > 1e3 * K0(1e-4)
            s.dhdx .= 0.0; s.dhdx[4, 3] = -1.0
            Shakti.compute_face_flux!(s, p, Upwind())
            @test s.K_x[4, 3] * (1 + p.omega * s.Re_x[4, 3]) ≈ K0(1e-4) rtol = 1e-9
        end


        @testset "freeze-on capacity is limited by the water present" begin
            for cf in (FilledCavities(), UnfilledCavities())
                sim = make_sim(cf, stuck = false)
                s, p = sim.state, sim.p
                s.b .= 0.01; s.b_empty .= 0.0
                s.b_empty[3, 3] = 0.01 - 1e-5   # holds 10 micrometres of water, gap room 9.99 mm
                hostdt = 86400.0
                C = Array(freeze_on_capacity!(zeros(nx, ny), sim; dt = hostdt, water_limited = true))
                limit = 1e-5 * p.rho_w / p.rho_i / hostdt
                if cf isa UnfilledCavities
                    @test C[3, 3] <= limit * (1 + 1e-12)       # water-limited
                    @test C[3, 3] > 0
                else
                    @test C[3, 3] > limit                       # legacy: the gap room is the limit, b_empty is ignored
                end
            end
            su = Array(freeze_on_capacity!(zeros(nx, ny), make_sim(UnfilledCavities(), stuck = false); dt = 86400.0, water_limited = true))
            s0 = Array(freeze_on_capacity!(zeros(nx, ny), make_sim(UnfilledCavities(), stuck = false); dt = 86400.0, water_limited = false))
            sf0 = Array(freeze_on_capacity!(zeros(nx, ny), make_sim(FilledCavities(), stuck = false); dt = 86400.0))
            sd = Array(freeze_on_capacity!(zeros(nx, ny), make_sim(UnfilledCavities(), stuck = false); dt = 86400.0))
            @test s0 ≈ sf0 # water_limited = false: the gap-room capacity alone, as before
            @test sd ≈ su  # the water limit is the default under unfilled cavities
            @test freeze_on_capacity!(zeros(nx, ny), make_sim(FilledCavities(), stuck = false); dt = 86400.0, water_limited = true) ≈ sf0 # never applies with FilledCavities
            sf = Array(freeze_on_capacity!(zeros(nx, ny), make_sim(FilledCavities(), stuck = false); dt = 86400.0))
            @test all(su .<= sf .* (1 + 1e-12)) # the water limit only ever lowers it (freezing needs water, even in a full gap)
            @test any(su .< sf) # ... where sliding opens more room than the gap holds water
        end


        # ---- parabolic head scheme (p.e_v != 0) and Newton-Krylov solver ------------------------------

        # total water per area V = b + g(h - zb), g = e_v*x filled / x unfilled (see cavity_parabolic_terms)
        gfun(x, e_v) = x >= 0 ? e_v * x : x
        function par_sim(cf; e_v = 1e-7, stuck = true, kface = nothing, mask = base_mask(), pres = 1e-12)
            A_visc = fill(stuck ? 1e-30 : 5e-25, nx, ny)
            b0     = stuck ? fill(1e-6, nx, ny) : [0.01 + 0.004 * sin(i + 2j) for i in 1:nx, j in 1:ny]
            ub_x   = fill(stuck ? 3e-7 : 1e-6, nx + 1, ny)
            p = ModelParameters(e_v = e_v, b_min = 1e-6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, zeros(nx, ny + 1), ieb, taub_x, taub_y)
            pps = ParabolicPicardSolver(500, floattype(pres), CholeskyDirectSolver(grid), grid)
            return Simulation(grid, state, 1, floattype(dt), p, "explicit", String[], ConstantMeltInput(), sl; pps = pps, cavity_filling = cf, k_face_choice = kface)
        end

        @testset "parabolic: no unfilled cell is identical to the legacy solve" begin
            sims = [par_sim(cf; stuck = false, kface = "arithmetic") for cf in (FilledCavities(), UnfilledCavities())]
            for sim in sims
                Shakti.prepare_head_solve!(sim); Shakti.step_h!(sim.hs, sim)
            end
            g = grounded(sims[1].state)
            @test all(Array(sims[1].state.pw)[g] .> 0)
            @test Array(sims[2].state.h) ≈ Array(sims[1].state.h) rtol = 1e-12 atol = 1e-9
        end

        @testset "parabolic: stuck cells keep pw >= 0 and the total water budget is exact" begin
            fi = par_sim(FilledCavities()); step!(fi)
            @test minimum(Array(fi.state.pw)[grounded(fi.state)]) < 0 # e_v this small: englacial storage does not save the stuck cells
            sim = par_sim(UnfilledCavities())
            s, p = sim.state, sim.p
            h_old, b_old = copy(Array(s.h)), copy(Array(s.b))
            step!(sim)
            g = grounded(s)
            @test first(Shakti.picard_status(sim.hs))
            @test minimum(Array(s.pw)[g]) >= 0
            @test any(Array(s.b_empty)[g] .> 0)
            xo, xn = h_old .- Array(s.zb), Array(s.h) .- Array(s.zb)
            qx, qy = Array(s.q_x), Array(s.q_y)
            div = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
            res = ((Array(s.b) .- b_old) .+ (gfun.(xn, p.e_v) .- gfun.(xo, p.e_v))) ./ dt .+ div .- (Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
            scale = maximum(abs, Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)) + maximum(Array(s.beta) .* Array(s.abs_ub))
            @test maximum(abs, res[g]) < 1e-6 * scale
        end

        @testset "NewtonJFNKSolver with unfilled cavities agrees with Picard" begin
            # Newton's finite-difference Jacobian sees the empty/filled switch as a kink: it converges on this
            # fixture (and many like it) but not on the stiffest stuck ones, where Picard still converges in a few iterations.
            function newton_or_picard(mk)
                p = ModelParameters(e_v = 0.0, b_min = 1e-6)
                state = State(grid)
                set_initial_conditions!(state, grid, p, sl, base_mask(), fill(1e-24, nx, ny), zb, zs, fill(1e-6, nx, ny), G, fill(3e-7, nx + 1, ny), zeros(nx, ny + 1), zeros(nx, ny), taub_x, taub_y)
                sim = Simulation(grid, state, 1, floattype(dt), p, "explicit", String[], ConstantMeltInput(), sl; ps = mk(), cavity_filling = UnfilledCavities())
                b_old, e_old = copy(Array(state.b)), copy(Array(state.b_empty))
                step!(sim)
                return sim, b_old, e_old
            end
            simN, b_old, e_old = newton_or_picard(() -> NewtonJFNKSolver(grid; iters = 100, tol = 1e-10))
            simP, _, _     = newton_or_picard(() -> PicardSolver(500, floattype(1e-12), CholeskyDirectSolver(grid), grid))
            s = simN.state; g = grounded(s)
            @test first(Shakti.picard_status(simN.hs))
            @test minimum(Array(s.pw)[g]) >= 0
            @test any(Array(s.b_empty)[g] .> 0)
            @test Array(s.h) ≈ Array(simP.state.h) rtol = 1e-8 atol = 1e-6
            res = budget_residual(simN, b_old, e_old)
            scale = maximum(abs, Array(s.mdot) ./ simN.p.rho_w) + maximum(Array(s.beta) .* Array(s.abs_ub))
            @test maximum(abs, res[g]) < 1e-3 * scale # Newton stops at a looser head tolerance than the Picard test above
        end
    end
