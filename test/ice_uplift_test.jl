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
        # The face conductances use the water depth at the start of the step (the water W), so dt must resolve
        # how fast an uplift layer drains: at dt = 3 h the moulin cell's 0.7 m layer (K up 1e8-fold) empties
        # within the step and Picard cycles; at 30 min it converges and the budget closes to round-off.
        dt     = 1800.0

        # a 3 m^3/s moulin into a 1 mm gap (200 m cells): far more water than the sheet can carry at overburden
        function make_sim(; up = NoUplift(), cf = FilledCavities(), gap = ExplicitGapScheme(), e_v = 0.0, mask = base_mask(), moulin = 3.0, dt = dt, iters = 500, ts = _ -> FixedTimeStep())
            A_visc = fill(5e-25, nx, ny)
            b0     = fill(1e-3, nx, ny)
            ub_x   = fill(1e-6, nx + 1, ny); ub_y = zeros(nx, ny + 1)
            ieb    = zeros(nx, ny); ieb[3, 3] = moulin / (grid.dx * grid.dy)
            p = ModelParameters(e_v = e_v, b_min = 1e-6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ls = CholeskyDirectSolver(grid)
            return Simulation(grid, state, 1, floattype(dt), p, ConstantMeltInput(), sl; gap_scheme = gap, k_face_choice = "upwind", cavity_filling = cf, uplift = up,
                              ps = PicardSolver(iters, floattype(1e-12), ls, grid), timestep_scheme = ts(state))
        end
        grounded(s) = Array(s.mask) .== GROUNDED
        h_o(s, p) = Array(s.zb) .+ Array(s.po) ./ (p.rho_w * p.g)

        function budget_residual(sim, W_old)
            s, p = sim.state, sim.p
            qx, qy = Array(s.q_x), Array(s.q_y)
            div = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
            dW = (Array(s.W) .- W_old) ./ dt
            return dW .+ div .- (Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
        end
        water(s) = Array(s.W)

        @testset "legacy: the moulin drives pw above the overburden" begin
            sim = make_sim()
            step!(sim)
            g = grounded(sim.state)
            @test maximum((Array(sim.state.pw) .- Array(sim.state.po))[g]) > 0
            @test sim.up isa NoUplift
            @test sim.hp isa SheetTopPotential && make_sim(; up = IceUplift()).hp isa SheetTopPotential # the default potential
            @test all(iszero, Array(water_above_gap(sim.state))) && all(iszero, Array(sim.state.stor_up))
        end

        @testset "opening by sliding stops where the ice floats above the bumps" begin
            sim = make_sim(; up = IceUplift())
            s, p = sim.state, sim.p
            b = Array(s.b)
            @test all(==(p.br), Array(s.bump)) # no uplift yet: the full bump height, bit for bit
            bu = zeros(nx, ny); bu[3, 3] = 0.5 * p.br; bu[4, 3] = 2 * p.br
            copyto!(s.W, b .+ bu) # the uplift layer is the water beyond the gap where the head is above the overburden
            hh = Array(s.h); ho = Array(s.zb) .+ Array(s.po) ./ (p.rho_w * p.g)
            hh[3, 3] = ho[3, 3] + 1; hh[4, 3] = ho[4, 3] + 1
            copyto!(s.h, hh)
            @test Array(s.stor_up)[3, 3] > 0 && Array(s.stor_up)[4, 3] > 0
            compute_beta!(s, p, Shakti.WithOpenBySliding())
            bump, beta = Array(s.bump), Array(s.beta)
            @test bump[3, 3] ≈ 0.5 * p.br && bump[4, 3] == 0 && bump[2, 2] == p.br
            @test beta[3, 3] ≈ max(0, (0.5 * p.br - b[3, 3]) / p.lr) # the ice base is at the uplift layer W - b
            @test beta[4, 3] == 0                                     # floating above every bump: no opening
            @test beta[2, 2] ≈ (p.br - b[2, 2]) / p.lr
            # the fully implicit gap the head rows assume (cell_gap) has no sliding opening either: at N = 0 with no melt
            # the gap of the floating cell stays, while with the full bump height it would open
            gu = Shakti.SchemeGapUpdate{Shakti.FullyImplicitGapScheme, Shakti.StandardCreep, Float64}(dt, p.br, p.lr, p.b_c, false)
            ub = 1e-6
            @test Shakti.gap_update(Shakti.cell_gap(gu, bump[4, 3]), b[4, 3], 0.0, beta[4, 3], ub, 0.0, p.rho_i) ≈ b[4, 3]
            @test Shakti.gap_update(gu, b[4, 3], 0.0, beta[2, 2], ub, 0.0, p.rho_i) > b[4, 3]
            @test Shakti.cell_gap(ExplicitGapCoupling(), 0.0) isa ExplicitGapCoupling
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
                    @test all(Array(water_above_gap(s))[up] .> 0)
                    @test all(Array(water_above_gap(s))[g .& .!up] .<= 1e-8 * maximum(Array(s.W))) # the Picard tolerance of the budget only (4e-10 m at the moulin under upwind)
                    res = budget_residual(sim, W_old)
                    scale = maximum(abs, Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
                    @test maximum(abs, res[g]) < 1e-6 * scale
                end
                # head-based layer: W - b = h - h_o where lifted (Filled; Unfilled takes it from the budget, equal at convergence)
                g = grounded(s); up = g .& (Array(s.h) .> h_o(s, p))
                @test Array(water_above_gap(s))[up] ≈ (Array(s.h) .- h_o(s, p))[up] rtol = 1e-6
            end
        end

        @testset "uplift row is continuous with the filled row at h = h_o" begin
            q = ModelParameters()
            b, mdot, beta, ub, A, lc, zb0, po0 = 1e-3, 0.0, 0.05, 1e-6, 5e-25, 1e-3, 10.0, 4e6
            ho = zb0 + po0 / (q.rho_w * q.g)
            terms(h, N) = Shakti.cavity_head_terms(b, q.b_min, q.b_max, mdot, beta, ub, A, N, lc, h, zb0, po0, 0.0,
                                                    q.rho_w, q.rho_i, q.g, q.n, q.n_minus_1_exp, 1 / dt, b)
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
            @test water_above_gap(s)[3, 3] > 0
            res = budget_residual(sim, W_old)
            @test abs(res[3, 3]) < 1e-9 * abs(Array(s.ieb)[3, 3])
        end

        @testset "AdaptiveTimeStep: a step that does not converge is retried at half the dt, fast steps grow it" begin
            # 80 Picard iterations: step 2 does not converge at 3 h and is taken at 1.5 h, then dt grows back to 3 h
            day = 86400.0
            adaptive(st) = AdaptiveTimeStep(grid, st; dt_min = 600.0, dt_max = 3 * 3600.0, target_time = day, fast_iters = 40, grow_after = 1)
            sim = make_sim(; up = IceUplift(), cf = UnfilledCavities(), dt = 3 * 3600.0, iters = 80, ts = adaptive)
            ts = sim.ts
            dts, conv = Float64[], Bool[]
            while sim.total_time[] < day
                step!(sim); push!(dts, sim.dt[]); push!(conv, first(Shakti.picard_status(sim.hs))); sim.total_time[] += sim.dt[]
            end
            @test ts.rejected[] > 0 && all(conv)
            @test minimum(dts) < 3 * 3600.0 && maximum(dts[argmin(dts):end - 1]) == 3 * 3600.0 # halved, then grown back to dt_max
            @test sim.total_time[] == day
            # only the head was redone: the same steps at fixed dt give the same state bit for bit
            ref = make_sim(; up = IceUplift(), cf = UnfilledCavities(), iters = 80)
            for dt_k in dts
                ref.dt[] = dt_k; step!(ref)
            end
            @test Array(ref.state.h) == Array(sim.state.h) && Array(ref.state.W) == Array(sim.state.W) && Array(ref.state.b) == Array(sim.state.b)
            # a step at dt_min is kept whether it converged or not
            stuck = make_sim(; up = IceUplift(), cf = UnfilledCavities(), dt = 3600.0, iters = 2,
                             ts = st -> AdaptiveTimeStep(grid, st; dt_min = 1800.0, dt_max = 3600.0, target_time = day))
            step!(stuck)
            @test stuck.dt[] == 1800.0 && stuck.ts.rejected[] == 1 && !first(Shakti.picard_status(stuck.hs))
            # the optional rate cap rates only the gaps above b_min
            capped = make_sim(; ts = st -> AdaptiveTimeStep(grid, st; dt_min = 1.0, dt_max = day, target_time = day, safety_factor = 0.3))
            s = capped.state
            Shakti.update_dt!(capped)
            rate = Array(capped.ts.rate_field)
            @test isfinite(rate[3, 3]) && capped.dt[] < day
            b = Array(s.b); b[3, 3] = capped.p.b_min; copyto!(s.b, b)
            Shakti.update_dt!(capped)
            @test Array(capped.ts.rate_field)[3, 3] == -Inf
            # the controller is checkpointed with the state
            mktempdir() do dir
                ck = joinpath(dir, "ck.jld2")
                ts.dt_ctrl[] = 1234.0; ts.fast_count[] = 2
                Shakti.save_checkpoint(ck, sim, 1)
                other = make_sim(; up = IceUplift(), cf = UnfilledCavities(), ts = adaptive)
                Shakti.load_checkpoint!(other, ck)
                @test other.ts.dt_ctrl[] == 1234.0 && other.ts.fast_count[] == 2
            end
        end

        @testset "parabolic scheme (e_v > 0): pw <= po" begin
            # e_v = 1e-4: the englacial storage below the overburden (e_v*p_o/(rho_w g), 5 cm) fills within the first steps
            sim = make_sim(; up = IceUplift(), e_v = 1e-4, gap = FullyImplicitGapScheme())
            s = sim.state
            for _ in 1:4
                step!(sim)
            end
            g = grounded(s)
            @test first(Shakti.picard_status(sim.hs))
            @test maximum((Array(s.pw) .- Array(s.po))[g]) <= 1e-9 * maximum(Array(s.po))
            @test any(Array(water_above_gap(s))[g] .> 0)
        end

    end
