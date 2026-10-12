    @testset "ImplicitUpwind (water depth at the end of the step, Newton in W)" begin

        sl = RegularizedCoulombSlidingLaw(0.25)
        ub = 30.0 / (365 * 86400)

        # 1-D bed ramp after Schoof, Hewitt & Werder (2012) Fig. 7, 40 x 3 cells of 100 m: sliding over the bumps opens the
        # gap faster than the melt fills it in the upper part (free surface, pw = 0), the lower part is filled; LAND margin
        # in the last column. Melt terms off, so the melt input m is the only source and mdot does not depend on the head.
        function ramp(; kface = "implicit_upwind", solver = :auto, dt = 3600.0, m_mmd = 5.0, omega = 1e-4, e_v = 0.0,
                        cf = UnfilledCavities(), up = NoUplift(), hp = BedPotential(), gap = "implicit", gc = ExplicitGapCoupling())
            nx, ny = 40, 3
            grid = Grid(nx, ny, 3.9e3, 200.0)
            x = Array(grid.x)
            zb = repeat(80.0 .* (1 .- x ./ x[end]), 1, ny)
            H  = repeat([400.0 * sqrt(max(0.0, 1 - xi / x[end])) for xi in x], 1, ny)
            mask = fill(GROUNDED, nx, ny); mask[end, :] .= LAND
            p = ModelParameters(; br = 0.1, lr = 2.0, b_min = 1e-6, e_v = e_v, omega = omega,
                                mdot_includes_G = false, mdot_includes_frictional = false, mdot_includes_potential = false,
                                mdot_includes_sensible = false, mdot_includes_qT = false, limit_freeze_on = false)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, fill(5e-25, nx, ny), zb, zb .+ H, fill(0.005, nx, ny), zeros(nx, ny),
                                    fill(ub, nx + 1, ny), zeros(nx, ny + 1), fill(m_mmd / 1000 / 86400, nx, ny), zeros(nx + 1, ny), zeros(nx, ny + 1))
            solver === :auto && (solver = kface == "implicit_upwind" ? :lu : :cholesky)
            ls = solver === :lu ? LUDirectSolver(grid) : CholeskyDirectSolver(grid)
            kw = e_v == 0 ? (; ps = PicardSolver(200, floattype(1e-11), ls, grid)) : (; pps = ParabolicPicardSolver(200, floattype(1e-11), ls, grid))
            return Simulation(grid, state, 1, floattype(dt), p, gap, String[], ConstantMeltInput(), sl;
                              k_face_choice = kface, cavity_filling = cf, uplift = up, hydraulic_potential = hp, gap_coupling = gc, kw...)
        end
        grounded(sim) = Array(sim.state.mask) .== GROUNDED
        unfilled(sim) = grounded(sim) .& (Array(sim.state.h) .< Array(sim.state.zb))
        water(s) = Array(s.b) .- Array(s.b_empty) .+ Array(s.b_uplift)
        function budget_residual(sim, W_old)
            s, p, g = sim.state, sim.p, sim.grid
            qx, qy = Array(s.q_x), Array(s.q_y)
            div = [(qx[i+1, j] - qx[i, j]) / g.dx + (qy[i, j+1] - qy[i, j]) / g.dy for i in 1:g.nx, j in 1:g.ny]
            return (water(s) .- W_old) ./ sim.dt[] .+ div .- (Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
        end

        @testset "no free water depth: the heads of Upwind" begin
            a = ramp(kface = "upwind", cf = FilledCavities())
            b = ramp(kface = "implicit_upwind", cf = FilledCavities())
            for _ in 1:3
                step!(a); step!(b)
            end
            @test Array(b.state.h) ≈ Array(a.state.h) rtol = 1e-9
            @test all(iszero, Array(b.state.Jw_x)) && all(iszero, Array(b.state.Jw_y))
        end

        @testset "Newton coefficient is d|q|/dW of the donor (omega = $omega)" for omega in (0.0, 1e-4, 0.8)
            sim = ramp(omega = omega, m_mmd = omega > 0.1 ? 0.02 : 5.0) # rough flow (omega = 0.8) conducts ~100 times less: less input for a free surface
            for _ in 1:3
                step!(sim)
            end
            s, p = sim.state, sim.p
            Shakti.prepare_head_solve!(sim)
            k = argmax(abs.(Array(s.Jw_x)))
            i, j = Tuple(k)
            J = Array(s.Jw_x)[i, j]
            @test abs(J) > 0
            d = J > 0 ? i : i - 1 # the donor cell
            q0 = abs(Array(s.q_x)[i, j])
            δ = 1e-7 * Array(s.b_w)[d, j]
            s.b[d, j] += δ # W = b + (h - zb) in the unfilled donor: moves W alone, the gradient stays
            Shakti.compute_face_flux!(s, p, sim.kfs)
            q1 = abs(Array(s.q_x)[i, j])
            s.b[d, j] -= δ
            @test abs(J) ≈ (q1 - q0) / δ rtol = 1e-4
        end

        @testset "head matrix is the Jacobian of the row residual (laminar, $(nameof(typeof(gc))))" for gc in (ExplicitGapCoupling(), ConsistentGapCoupling())
            # laminar: the conductance depends on the head only through the water depth (more input: laminar flow conducts more,
            # and both regimes are wanted). Under ConsistentGapCoupling the filled cells' water depth is the gap scheme's G(N(h)) too.
            sim = ramp(omega = 0.0, m_mmd = 50.0, gc = gc)
            for _ in 1:3
                step!(sim)
            end
            s, g, p = sim.state, sim.grid, sim.p
            h0 = copy(Array(s.h)) # the converged head (the extrapolated guess may sit exactly on a regime boundary, a kink of the residual)
            Shakti.prepare_head_solve!(sim)
            sals = sim.hs.ps.ls.sals
            # The rows assembled at h hold the Newton terms J*(h - h^k) at h^k = h, which cancel: M(h)*h - rhs(h) is
            # the residual of the implicit (end-of-step water depth) equations at h.
            gct = Shakti.gap_coupling_terms(sim)
            function residual(h)
                s.h .= h
                Shakti.reset_donors!(sim.kfs, s) # every evaluation from the higher-potential donors (no flip counting across evaluations)
                refresh_head_dependents!(s, g, p, sim.mt, sim.kfs, sim.sl; gc = gct)
                Shakti.update_SALS_elliptic!(sals, s, g, p, sim.kfs, NoDiffusion(), gct)
                return sals.M * vec(h) .- sals.rhs
            end
            residual(h0)
            M0 = Matrix(sals.M)
            gr = findall(vec(grounded(sim)))
            @test any(unfilled(sim)) && any(grounded(sim) .& .!unfilled(sim)) # both regimes present
            Jfd = zeros(length(gr), length(gr))
            for (c, jj) in enumerate(gr)
                δ = 1e-6
                hp = copy(h0); hp[jj] += δ
                hm = copy(h0); hm[jj] -= δ
                Jfd[:, c] = ((residual(hp) .- residual(hm)) ./ (2δ))[gr]
            end
            residual(h0)
            @test maximum(abs, M0[gr, gr] .- Jfd) < 1e-5 * maximum(abs, M0[gr, gr]) # (the gap slopes are central differences)
            @test !issymmetric(M0[gr, gr]) # the Newton terms are in
        end

        @testset "dt = 1 day: converges every step, pw >= 0, budget closes, steady state of dt = 2 h" begin
            a = ramp(dt = 86400.0)
            b = ramp(dt = 7200.0)
            for k in 1:300
                W_old = water(a.state)
                step!(a)
                @test first(Shakti.picard_status(a.hs))
                k == 300 || continue
                g = grounded(a)
                @test minimum(Array(a.state.pw)[g]) >= 0
                res = budget_residual(a, W_old)
                @test maximum(abs, res[g]) < 1e-6 * maximum(abs, Array(a.state.ieb)[g])
            end
            for _ in 1:3600
                step!(b)
            end
            g = grounded(a)
            @test any(unfilled(a))
            @test Array(a.state.h)[g] ≈ Array(b.state.h)[g] rtol = 1e-6
            @test Array(a.state.b_w)[g] ≈ Array(b.state.b_w)[g] rtol = 1e-3
        end

        @testset "gap scheme $gap at dt = 1 day" for gap in ("explicit", "implicit", "fully_implicit")
            sim = ramp(dt = 86400.0, gap = gap)
            for _ in 1:20
                step!(sim)
                @test first(Shakti.picard_status(sim.hs))
            end
            @test all(isfinite, Array(sim.state.h))
        end

        @testset "water-surface potential and parabolic scheme at dt = 1 day" begin
            for kw in ((; hp = SheetTopPotential()), (; e_v = 1e-4))
                sim = ramp(; dt = 86400.0, kw...)
                for _ in 1:20
                    step!(sim)
                    @test first(Shakti.picard_status(sim.hs))
                end
                g = grounded(sim)
                @test minimum(Array(sim.state.pw)[g]) >= 0
            end
        end

        @testset "uplift: a 3 m^3/s moulin at dt = 3 h and 1 day" begin
            # The moulin fixture of ice_uplift_test.jl, where the start-of-step water depth needs dt <= 30 min (and even there the
            # moulin cell lifts and drops on alternate steps). The uplift decays as sliding and melt open the gap at N = 0. At dt = 1 day
            # the explicit gap scheme itself fails (creep*dt ~ 4 collapses the gap to b_min), so that step uses the fully implicit one.
            # The default gap coupling of implicit_upwind (ConsistentGapCoupling) is used: end-of-step gap in the conductance too.
            nx, ny = 6, 6
            grid = Grid(nx, ny, 1e3, 1e3)
            mask = fill(GROUNDED, nx, ny); mask[end, :] .= OCEAN; mask[1, :] .= LAND; mask[:, 1] .= OTHER_BASIN
            zb = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
            for (dt, gap) in ((3 * 3600.0, "explicit"), (86400.0, "fully_implicit"))
                p = ModelParameters(b_min = 1e-6)
                state = State(grid)
                ieb = zeros(nx, ny); ieb[3, 3] = 3.0 / (grid.dx * grid.dy)
                set_initial_conditions!(state, grid, p, sl, mask, fill(5e-25, nx, ny), zb, zb .+ 500.0, fill(1e-3, nx, ny), fill(0.06, nx, ny),
                                        fill(1e-6, nx + 1, ny), zeros(nx, ny + 1), ieb, zeros(nx + 1, ny), zeros(nx, ny + 1))
                sim = Simulation(grid, state, 1, floattype(dt), p, gap, String[], ConstantMeltInput(), sl;
                                 ps = PicardSolver(500, floattype(1e-10), LUDirectSolver(grid), grid), k_face_choice = "implicit_upwind",
                                 cavity_filling = UnfilledCavities(), uplift = IceUplift())
                s = sim.state
                for k in 1:4
                    W_old = water(s)
                    step!(sim)
                    g = Array(s.mask) .== GROUNDED
                    @test first(Shakti.picard_status(sim.hs))
                    @test maximum((Array(s.pw) .- Array(s.po))[g]) <= 1e-9 * maximum(Array(s.po))
                    k == 1 && dt < 86400 && @test Array(s.b_uplift)[3, 3] > 0 # the moulin cell lifts the ice (at dt = 1 day the gap has opened enough within the step)
                    res = budget_residual(sim, W_old)
                    @test maximum(abs, res[g]) < 1e-9 * maximum(abs, Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
                end
            end
        end

        @testset "ConsistentGapCoupling: the head rows store what the gap scheme opens" begin
            # Filled cavities (the original equations), implicit gap scheme, dt = 1 day: creep*dt ~ 2 on the ramp. The head rows
            # of ExplicitGapCoupling assume the explicit gap change, so water is created or lost; with ConsistentGapCoupling the
            # water balance of every cell closes.
            gap_balance(sim, b_old) = begin
                s, g, p = sim.state, sim.grid, sim.p
                qx, qy = Array(s.q_x), Array(s.q_y)
                div = [(qx[i+1, j] - qx[i, j]) / g.dx + (qy[i, j+1] - qy[i, j]) / g.dy for i in 1:g.nx, j in 1:g.ny]
                res = (Array(s.b) .- b_old) ./ sim.dt[] .+ div .- (Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
                maximum(abs, res[grounded(sim)]) / maximum(abs, Array(s.ieb)[grounded(sim)])
            end
            for (gc, expect_closed) in ((ExplicitGapCoupling(), false), (ConsistentGapCoupling(), true))
                sim = ramp(kface = "upwind", cf = FilledCavities(), dt = 86400.0, gc = gc)
                worst = 0.0
                for _ in 1:10
                    b_old = copy(Array(sim.state.b))
                    step!(sim)
                    @test first(Shakti.picard_status(sim.hs))
                    worst = max(worst, gap_balance(sim, b_old))
                end
                expect_closed ? (@test worst < 1e-6) : (@test worst > 1e-2)
            end
            # with the explicit gap scheme the two couplings differ only by the b_min/b_max clamp (inactive here)
            a = ramp(kface = "upwind", cf = FilledCavities(), gap = "explicit", gc = ExplicitGapCoupling())
            c = ramp(kface = "upwind", cf = FilledCavities(), gap = "explicit", gc = ConsistentGapCoupling())
            for _ in 1:3
                step!(a); step!(c)
            end
            @test Array(c.state.h) ≈ Array(a.state.h) rtol = 1e-6
            # unfilled cavities, implicit water depth, every gap scheme (the explicit one at dt = 3 h: at 1 day creep*dt ~ 2 makes
            # the explicit gap update itself collapse to b_min): the empty volume stays within its limits (no water created or
            # lost by the bookkeeping) and the water budget closes
            for (gap, dt) in (("explicit", 3 * 3600.0), ("implicit", 86400.0), ("fully_implicit", 86400.0))
                sim = ramp(dt = dt, gap = gap, gc = ConsistentGapCoupling())
                for _ in 1:30
                    W_old = water(sim.state)
                    step!(sim)
                    @test first(Shakti.picard_status(sim.hs))
                    g = grounded(sim)
                    res = budget_residual(sim, W_old)
                    @test maximum(abs, res[g]) < 1e-6 * maximum(abs, Array(sim.state.ieb)[g])
                end
            end
        end

        @testset "regime-crossing and within-regime step limit" begin
            # One cell of the ramp (unfilled cavities + uplift, so both limits are active), iterate h_prev -> h_new, after the limit:
            sim = ramp(up = IceUplift())
            step!(sim)
            s = sim.state
            i, j = 10, 2
            z, bc = Array(s.zb)[i, j], Array(s.b)[i, j]
            ho = z + Array(s.po)[i, j] / (sim.p.rho_w * sim.p.g)
            ins = max(1e-6 * bc, 64 * eps(abs(z) + one(z)))
            function limited(hp, hn; dl = 0.0)
                dli = zeros(size(s.dh_it)); dli[i, j] = dl; copyto!(s.dh_it, dli) # the previous step of the cell (oscillation damping)
                h_prev = copy(Array(s.h)); h_prev[i, j] = hp
                h = copy(h_prev); h[i, j] = hn
                copyto!(s.h, h)
                Shakti.limit_regime_crossing!(sim.kfs, s, typeof(s.h)(h_prev), sim.p)
                return Array(s.h)[i, j]
            end
            @test ho > z + 10 && bc > 4e-3
            @test limited(z + 1, z - 5) ≈ z - ins                   # filled -> unfilled: just below the bed
            @test limited(z - 1e-3, z + 5) == z                      # unfilled -> filled: at the bed
            @test limited(z + 1, ho + 5) ≈ ho + ins                  # filled -> uplifted: just above the overburden head
            @test limited(ho + 1e-3, z + 1) == ho                    # uplifted -> filled: at the overburden head
            @test limited(z - ins, z - 1000) ≈ z - ins - 0.75 * bc  # unfilled: from the bed, 3/4 of the gap
            @test limited(z - 2e-3, z - 1000) ≈ z - 2e-3 - max(0.75 * bc, 6e-3)
            @test limited(z - 50, z - 1000) ≈ z - 200               # below the bed: the depth grows at most 4-fold
            @test limited(ho + 1e-2, ho + 1e4) ≈ ho + 1e-2 + 3 * (bc + 1e-2) # uplift layer grows at most 4-fold
            @test limited(z - 2e-3, z - 2.1e-3) == z - 2.1e-3        # small steps are untouched
            @test limited(ho + 1e-2, ho + 2e-2) == ho + 2e-2
            @test limited(z - 2e-3, z - 2e-3) == z - 2e-3            # a fixed point stays one
            hm, δ = (z + ho) / 2, (ho - z) / 10                     # filled, steps that stay filled
            @test limited(hm, hm + δ; dl = -0.4δ) ≈ hm + 0.2δ         # a reversal without contraction: half the previous step
            @test limited(hm, hm + 0.1δ; dl = -0.4δ) ≈ hm + 0.1δ      # a contracting reversal is untouched
            @test limited(hm, hm + δ; dl = 0.4δ) ≈ hm + δ             # no reversal: untouched
            @test Array(s.dh_it)[i, j] ≈ δ                              # the limited step is kept for the next iteration
        end

        @testset "face donor: higher-potential cell, frozen after MAX_DONOR_FLIPS flips" begin
            nd(d, dh) = Shakti.next_donor(d, dh, 4)
            @test nd(0.0, 2.0) == 1 && nd(0.0, -2.0) == -1 && nd(0.0, 0.0) == 0  # not set: the higher-potential cell
            @test nd(1.0, 3.0) == 1 && nd(1.0, 0.0) == 1                         # same direction or no flow: kept
            d = 1.0
            for k in 1:4
                d = nd(d, isodd(k) ? -1.0 : 1.0)
                @test sign(d) == (isodd(k) ? -1 : 1) && abs(d) == k + 1          # each reversal switches donor and counts
            end
            @test nd(d, -1.0) == d                                               # after 4 flips the donor stays
            # a step on the ramp starts with every donor reset, and the steady result is that of Upwind's donors
            sim = ramp(up = IceUplift())
            step!(sim)
            dx_ = Array(sim.state.don_x); dh = Array(sim.state.dhdx)
            @test all(sign.(dx_[dh .!= 0]) .== sign.(dh[dh .!= 0]))
        end

        @testset "needs a solver for non-symmetric matrices" begin
            @test_throws ErrorException ramp(kface = "implicit_upwind", solver = :cholesky)
        end

    end
