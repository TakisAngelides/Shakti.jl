    @testset "Melt in the gap opening (gap_opening_melt)" begin

        # 12 x 5 cells of 500 m: GROUNDED interior draining to an OCEAN edge, LAND and OTHER_BASIN on the others,
        # a bed sloping down towards the ocean under ice of constant thickness H.
        YR = 365.25 * 86400
        nx, ny = 12, 5
        grid = Grid(nx, ny, 500.0, 500.0)
        sl = RegularizedCoulombSlidingLaw(0.25)
        mask = fill(GROUNDED, nx, ny)
        mask[end, :] .= OCEAN
        mask[1, :]   .= LAND
        mask[:, 1]   .= OTHER_BASIN
        g = mask .== GROUNDED
        A_visc = fill(5e-25, nx, ny)
        zb     = repeat(reshape(-0.01 .* grid.x, nx, 1), 1, ny)
        b0     = fill(0.02, nx, ny)
        G      = fill(0.05, nx, ny)
        ieb    = zeros(nx, ny)
        taub_x = zeros(nx + 1, ny); taub_y = zeros(nx, ny + 1)
        ub_y   = zeros(nx, ny + 1)
        mean_g(a) = sum(a[g]) / count(g)

        function fresh_state(p; ub = 5.0, H = 600.0)
            s = State(grid)
            set_initial_conditions!(s, grid, p, sl, mask, A_visc, zb, zb .+ H, b0, G, fill(ub / YR, nx + 1, ny), ub_y, ieb, taub_x, taub_y)
            return s
        end
        params(mode; br = 0.1) = ModelParameters(e_v = 0.0, br = br, gap_opening_melt = mode)
        fresh_sim(mode; ub = 5.0, H = 600.0, br = 0.1, steps = 1, sheet = GlaDS2Sheet()) = begin
            p = params(mode; br)
            Simulation(grid, fresh_state(p; ub, H), steps, floattype(86400.0), p, ConstantMeltInput(), sl; sheet = sheet,
                       ps = PicardSolver(300, floattype(1e-9), LUDirectSolver(grid), grid))
        end
        # Steps a simulation, counting the steps whose Picard iteration did not converge; returns the last step's max |dN|.
        function run!(sim, steps)
            fails, dN = 0, NaN
            for _ in 1:steps
                Nold = copy(sim.state.N)
                step!(sim)
                first(Shakti.picard_status(sim.hs)) || (fails += 1)
                dN = maximum(abs.(sim.state.N .- Nold)[g])
            end
            return fails, dN
        end

        @testset "parameter and MeltTerms" begin
            @test ModelParameters().gap_opening_melt === :full
            @test_throws ArgumentError ModelParameters(gap_opening_melt = :uniform)
            @test Shakti.melt_terms(ModelParameters()) === MeltTerms{true, true, true, true, true, false}()
            @test Shakti.melt_terms(ModelParameters(gap_opening_melt = :exclude_uniform, mdot_includes_qT = false)) ===
                  MeltTerms{true, true, true, true, false, true}()
            @test MeltTerms{true, false, true, true, true}() === MeltTerms{true, false, true, true, true, false}() # five flags: :full
            @test fresh_sim(:exclude_uniform).mt === MeltTerms{true, true, true, true, true, true}()
        end

        @testset "compute_mdot!: melt_open, the contact weight and the freeze-on floor" begin
            p = ModelParameters(e_v = 0.0, br = 0.1)
            s = fresh_state(p)
            s.bump .= 0.1
            s.b[3, 3] = 0.02     # on its bumps: w_c = 0.8
            s.b[4, 3] = 0.15     # above its bumps: w_c = 0
            s.b[5, 3] = 0.1      # just clear of them: w_c = 0
            s.q_T[6, 3] = 0.3    # q_T > G + Q_b: net uniform freezing
            s.bump[7, 3] = 0.0   # no bumps under the ice (uplift or br = 0): w_c = 0
            sx = deepcopy(s)
            compute_mdot!(s, p, MeltTerms{true, true, true, true, true, false}())
            compute_mdot!(sx, p, MeltTerms{true, true, true, true, true, true}())
            @test s.melt_open == s.mdot ./ p.rho_i          # :full opens with all of the melt, bit for bit
            @test sx.mdot == s.mdot                         # the water source is the full melt in both modes
            @test sx.Q_b == s.Q_b && sx.Q_diss == s.Q_diss && sx.Q_sens == s.Q_sens
            Qu = s.G .+ s.Q_b .- s.q_T
            wc = clamp.((s.bump .- s.b) ./ s.bump, 0, 1); wc[s.bump .== 0] .= 0
            @test sx.melt_open ≈ (s.mdot .- wc .* max.(Qu, 0) ./ p.L) ./ p.rho_i rtol = 1e-12
            @test 0 < Qu[3, 3] && sx.melt_open[3, 3] < s.melt_open[3, 3]   # uniform melt on the bumps opens nothing
            @test Qu[6, 3] < 0 && sx.melt_open[6, 3] == s.melt_open[6, 3]  # net uniform freezing still closes the gap
            for c in ((4, 3), (5, 3), (7, 3))
                @test sx.melt_open[c...] == s.melt_open[c...]              # no contacts: all of the melt opens the gap
            end
            # The floor is on the opening: a floor above the opening holds it there, and the water keeps the excluded melt
            # on top, so mdot >= mdot_open >= the floor.
            mx = wc[3, 3] * max(Qu[3, 3], 0) / p.L
            mo = sx.melt_open[3, 3] * p.rho_i
            sx.mdot_min[3, 3] = mo + mx / 2
            compute_mdot!(sx, p, MeltTerms{true, true, true, true, true, true}())
            @test sx.melt_open[3, 3] * p.rho_i ≈ mo + mx / 2 rtol = 1e-12
            @test sx.mdot[3, 3] ≈ mo + mx / 2 + mx rtol = 1e-12
            @test sx.mdot[3, 3] >= sx.melt_open[3, 3] * p.rho_i >= sx.mdot_min[3, 3] * (1 - 1e-12)
        end

        @testset "br = 0: :exclude_uniform is :full" begin
            f, x = fresh_sim(:full; br = 0.0), fresh_sim(:exclude_uniform; br = 0.0)
            run!(f, 5); run!(x, 5)
            for name in (:h, :N, :b, :W, :mdot, :melt_open, :q_x, :q_y)
                @test getfield(x.state, name) == getfield(f.state, name)
            end
        end

        # Slow sliding over bumps, b_r = 0.1 m, G = 0.05 W/m^2: the melt the host reads is the full melt in both modes
        # from the same state (only melt_open differs); once a head solve has run, the heads differ and so do the
        # dissipation and the frictional heat (taub follows N). The water budget closes in both modes; at steady state
        # the smaller opening leaves a lower N.
        @testset "slow sliding, u_b = $ub m/a" for (ub, dNmin) in ((1.0, 0.01), (10.0, 1e-5))
            sf, sx = fresh_state(params(:full); ub), fresh_state(params(:exclude_uniform); ub)
            @test sx.mdot == sf.mdot && sx.Q_b == sf.Q_b    # the melt of the initial state
            @test all(sx.melt_open[g] .< sf.melt_open[g])   # on its bumps everywhere: less of it opens the gap
            f, x = fresh_sim(:full; ub), fresh_sim(:exclude_uniform; ub)
            ff, dNf = run!(f, 400)
            fx, dNx = run!(x, 400)
            @test ff == 0 && fx == 0
            @test dNf < 1e-3 && dNx < 1e-3                 # [Pa] per day: at steady state
            for sim in (f, x)
                @test abs(sim.wb.total_created[]) < 1e-12 * sim.wb.total_input[]
                @test sim.wb.gross_created[] < 1e-12 * sim.wb.total_input[]
            end
            rel = (mean_g(x.state.N) - mean_g(f.state.N)) / mean_g(f.state.N)
            @test -0.1 < rel < -dNmin                      # lower N: 2.8 % at 1 m/a, 0.02 % at 10 m/a (the gap takes the rest)
            @test all(x.state.N[g] .< f.state.N[g])
        end

        # No sliding: under :exclude_uniform only the dissipation opens the gap, so creep must nearly stop and N falls
        # toward 0 where the flux is small. The solver must follow it without failing (no N floor or clamp here).
        @testset "no sliding: N falls toward 0 without failing" begin
            H = 300.0
            f, x = fresh_sim(:full; ub = 0.0, H), fresh_sim(:exclude_uniform; ub = 0.0, H)
            ff, _ = run!(f, 600)
            fx, _ = run!(x, 600)
            po = 910 * 9.81 * H
            @test ff == 0 && fx == 0
            @test all(isfinite, x.state.N) && all(isfinite, x.state.h)
            @test minimum(x.state.N[g]) < 0.05 * po < minimum(f.state.N[g])
            @test minimum(x.state.N[g]) >= 0               # GlaDS2Sheet: water pressure at most overburden
            @test abs(x.wb.total_created[]) < 1e-12 * x.wb.total_input[]
        end
    end
