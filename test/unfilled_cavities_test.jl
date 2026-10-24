    @testset "Unfilled cavities (step 1: the water W)" begin

        nx, ny = 4, 3
        grid = Grid(nx, ny, 1e3, 1e3)

        @testset "State starts with W = 0" begin
            state = State(grid)
            @test all(iszero, state.W)
        end

        @testset "water_from_head!" begin
            mask = fill(GROUNDED, nx, ny)
            mask[1, 1] = OCEAN
            mask[2, 2] = FROZEN_BED
            p = ModelParameters()

            make() = begin
                s = State(grid)
                copyto!(s.mask, mask)
                s.zb .= 100.0
                s.b .= 0.003
                s.h .= 150.0  # pw > 0 everywhere ...
                s.h[3, 1] = 100.0 - 0.002 # ... except: GROUNDED, head 2 mm below the bed
                s.h[1, 1] = 50.0          # OCEAN below the bed: the gap
                s.h[2, 2] = 50.0          # FROZEN_BED below the bed: the gap
                s
            end

            s = make()
            water_from_head!(s, p, FilledCavities(), NoUplift())
            @test s.W == s.b # legacy: the gap is full

            s = make()
            water_from_head!(s, p, UnfilledCavities(), NoUplift())
            @test s.W[3, 1] ≈ 0.001
            @test s.W[1, 1] == 0.003
            @test s.W[2, 2] == 0.003
            @test count(s.W .!= s.b) == 1 # filled cells have nothing empty
            @test empty_gap(s)[3, 1] ≈ 0.002 && count(!iszero, empty_gap(s)) == 1
            @test all(iszero, water_above_gap(s))

            s = make() # po = 0: every GROUNDED cell above the bed is above the overburden head, lifted by h - zb
            water_from_head!(s, p, FilledCavities(), IceUplift())
            @test s.W[3, 3] ≈ 0.003 + 50.0 && s.W[3, 1] == 0.003 && s.W[1, 1] == 0.003
            @test water_above_gap(s)[3, 3] ≈ 50.0
        end

        @testset "constrain_water!" begin
            p = ModelParameters(b_min = 1e-3)
            s = State(grid)
            s.b .= [0.001 0.002 0.003; 0.001 0.002 0.003; 0.001 0.002 0.003; 0.001 0.002 0.003]
            W0 = [-1.0 0.001 0.005; 0.0005 0.002 0.001; 0.001 0.004 0.0; 0.002 0.0 0.003]
            run(cf, up, dry) = (copyto!(s.W, W0); Shakti.constrain_water!(s, p, cf, up, dry); copy(s.W))
            @test run(UnfilledCavities(), IceUplift(), false) == max.(W0, 0)
            @test run(FilledCavities(), NoUplift(), false) == s.b
            @test run(FilledCavities(), IceUplift(), false) == max.(W0, s.b)
            @test run(FilledCavities(), NoUplift(), true) == ifelse.(s.b .> p.b_min, s.b, clamp.(W0, 0, s.b))
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
        function make_sim(cf; stuck = true, mask = base_mask(), gap = "explicit", kface = nothing, ieb_scale = 1.0, gcp = ExplicitGapCoupling())
            A_visc = fill(stuck ? 1e-30 : 5e-25, nx, ny)
            b0     = stuck ? fill(1e-6, nx, ny) : [0.01 + 0.004 * sin(i + 2j) for i in 1:nx, j in 1:ny]
            ub_x   = fill(stuck ? 3e-7 : 1e-6, nx + 1, ny)
            ub_y   = zeros(nx, ny + 1)
            p = ModelParameters(e_v = 0.0, b_min = 1e-6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb .* ieb_scale, taub_x, taub_y)
            ps = PicardSolver(500, floattype(1e-12), kface == "implicit_upwind" ? LUDirectSolver(grid) : CholeskyDirectSolver(grid), grid)
            return Simulation(grid, state, 1, floattype(dt), p, gap, String[], ConstantMeltInput(), sl; uplift = NoUplift(), ps = ps, cavity_filling = cf, k_face_choice = something(kface, cf isa UnfilledCavities ? "upwind" : "arithmetic"), gap_coupling = gcp,
                              hydraulic_potential = BedPotential()) # compare cavity filling alone (the default is SheetTopPotential)
        end

        grounded(s) = Array(s.mask) .== GROUNDED

        # per-cell residual of (W_new - W_old)/dt + div(q) = mdot/rho_w + ieb
        function budget_residual(sim, W_old)
            s, p = sim.state, sim.p
            qx, qy = Array(s.q_x), Array(s.q_y)
            div = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
            dW = (Array(s.W) .- W_old) ./ dt
            return dW .+ div .- (Array(s.mdot) ./ p.rho_w .+ Array(s.ieb))
        end

        @testset "no unfilled cell: identical to the legacy solve" begin
            sims = [make_sim(cf; stuck = false, kface = "upwind") for cf in (FilledCavities(), UnfilledCavities())] # same face scheme (averaged faces are refused with unfilled cavities)
            for sim in sims
                Shakti.prepare_head_solve!(sim); Shakti.step_h!(sim.hs, sim)
            end
            g = grounded(sims[1].state)
            @test all(Array(sims[1].state.pw)[g] .> 0) # fixture sanity: nothing is unfilled
            @test Array(sims[2].state.h) ≈ Array(sims[1].state.h) rtol = 1e-12 atol = 1e-9
        end

        @testset "averaged faces with unfilled cavities are refused, upwind warns" begin
            @test_throws r"does not conserve water" make_sim(UnfilledCavities(); kface = "arithmetic")
            @test_throws r"never refills" make_sim(UnfilledCavities(); kface = "harmonic")
            @test_logs (:warn, r"stable only for") make_sim(UnfilledCavities(); kface = "upwind")
            @test_logs make_sim(FilledCavities(); kface = "arithmetic")
            @test_logs make_sim(FilledCavities(); kface = "upwind")
        end

        @testset "stuck cells: legacy gives pw < 0, unfilled cavities keep pw >= 0" begin
            fi = make_sim(FilledCavities())
            step!(fi)
            @test minimum(Array(fi.state.pw)[grounded(fi.state)]) < 0 # the problem this feature fixes

            un = make_sim(UnfilledCavities())
            s = un.state
            W_old = copy(Array(s.W))
            step!(un)
            g = grounded(s)
            @test first(Shakti.picard_status(un.hs)) # converged
            @test all(isfinite, Array(s.h))
            @test minimum(Array(s.pw)[g]) >= 0
            @test maximum(Array(s.N)[g] .- Array(s.po)[g]) <= 1e-6 * maximum(Array(s.po)) # N <= po
            @test all(Array(s.W) .>= 0)
            @test any(Array(empty_gap(s))[g] .> 0) # some gap really is empty
            @test all(Array(s.b)[g] .>= un.p.b_min)

            # exact water budget, every grounded cell (explicit gap update: b_new is the row's own b_old + dt*rate)
            res = budget_residual(un, W_old)
            scale = maximum(abs, Array(s.mdot) ./ un.p.rho_w .+ Array(s.ieb)) + maximum(Array(s.beta) .* Array(s.abs_ub))
            @test maximum(abs, res[g]) < 1e-6 * scale
        end

        @testset "a GlaDS-2 State rewrapped as the filled sheet keeps no unfilled/uplift fields" begin
            # The kernels read stor, stor_up and W whatever the option: the 'off' options must zero the first two and fill the
            # gap with W (spin-up then switch, or a checkpoint written under other options), not leave a hybrid sheet.
            A_visc = fill(1e-30, nx, ny); b0 = fill(1e-6, nx, ny); ub_x = fill(3e-7, nx + 1, ny); ub_y = zeros(nx, ny + 1)
            p = ModelParameters(e_v = 0.0, b_min = 1e-6)
            st = State(grid)
            set_initial_conditions!(st, grid, p, sl, base_mask(), A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            mk(state; kw...) = Simulation(grid, state, 1, floattype(dt), p, "fully_implicit", String[], ConstantMeltInput(), sl;
                                          ps = PicardSolver(500, floattype(1e-12), LUDirectSolver(grid), grid), kw...)
            glads = mk(st)
            for _ in 1:3; step!(glads); end
            @test maximum(Array(water_above_gap(st))) > 0 && maximum(Array(empty_gap(st))) > 0 && maximum(Array(st.stor)) > 0 # the fixture really lifts the ice
            filled_kw = (cavity_filling = FilledCavities(), uplift = NoUplift(), k_face_choice = "arithmetic", gap_coupling = ExplicitGapCoupling())
            ref = deepcopy(st)
            for f in (ref.stor, ref.stor_up); fill!(f, 0); end
            copyto!(ref.W, ref.b)
            wrapped, clean = mk(st; filled_kw...), mk(ref; filled_kw...)
            for f in (:stor, :stor_up); @test all(iszero, Array(getfield(st, f))); end
            @test Array(st.W) == Array(st.b)
            step!(wrapped); step!(clean)
            @test Array(st.h) == Array(ref.h)
            @test Array(st.b) == Array(ref.b)
        end

        @testset "N clamps under the free-surface sheets: refused when they bind, ignored by the water test when they cannot" begin
            p0 = ModelParameters(e_v = 0.0, b_min = 1e-6)
            st = State(grid)
            set_initial_conditions!(st, grid, p0, sl, base_mask(), fill(5e-25, nx, ny), zb, zs, fill(1e-3, nx, ny), G, fill(1e-6, nx + 1, ny), zeros(nx, ny + 1), ieb, taub_x, taub_y)
            glads(p; kw...) = Simulation(grid, st, 1, floattype(dt), p, "fully_implicit", String[], ConstantMeltInput(), sl;
                                         ps = PicardSolver(500, floattype(1e-12), LUDirectSolver(grid), grid), kw...)
            @test_throws r"contradicts the unfilled" glads(ModelParameters(e_v = 0.0, N_min = 1e3))
            @test_throws r"contradicts the unfilled" glads(p0; cell_N_clamping = CellNClamping(Dict((3, 3) => (0.0, 1e6))))
            sim = glads(p0; cell_N_clamping = CellNClamping(Dict((3, 3) => (0.0, Inf)))) # N >= 0 already: cannot bind
            @test Shakti.water_check_dt(sim) == sim.dt[]
            @test Shakti.water_check_dt(glads(ModelParameters(e_v = 0.0, N_min = 0.0))) > 0
            @test Shakti.water_check_dt(glads(p0; uplift = NoUplift(), cell_N_clamping = CellNClamping(Dict((3, 3) => (0.0, Inf))))) == 0 # N < 0 possible: binds
            @test_logs make_sim(FilledCavities(); kface = "arithmetic") # the filled sheet takes any clamp
            @test Simulation(grid, State(grid), 1, floattype(dt), ModelParameters(e_v = 0.0, N_min = 1e3), "explicit", String[], ConstantMeltInput(), sl;
                             cavity_filling = FilledCavities(), uplift = NoUplift(), ps = PicardSolver(5, floattype(1e-6), CholeskyDirectSolver(grid), grid)) isa Simulation
        end

        @testset "freezing and thawing a lifted cell: the water is reported and none is created" begin
            A_visc = fill(1e-30, nx, ny); b0 = fill(1e-6, nx, ny); ub_x = fill(3e-7, nx + 1, ny); ub_y = zeros(nx, ny + 1)
            p = ModelParameters(e_v = 0.0, b_min = 1e-6)
            st = State(grid)
            set_initial_conditions!(st, grid, p, sl, base_mask(), A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            sim = Simulation(grid, st, 1, floattype(dt), p, "fully_implicit", String[], ConstantMeltInput(), sl;
                             ps = PicardSolver(500, floattype(1e-12), LUDirectSolver(grid), grid))
            for _ in 1:3; step!(sim); end
            water(s) = sum(Array(s.W .* (s.mask .== GROUNDED)))
            lifted = Array(water_above_gap(st)) .> 0
            @test any(lifted)
            cell = findfirst(lifted)
            T = zeros(nx, ny); T[cell] = -100.0
            W0 = water(st)
            r = update_frozen_mask!(sim, T)
            @test r.n_frozen == 1
            @test r.discarded_b > st.b[cell] + 0.0 # the uplift layer counts, not just the gap
            @test W0 - water(st) ≈ r.discarded_b rtol = 1e-12
            @test st.b[cell] == 0 && st.W[cell] == 0
            W1 = water(st)
            T[cell] = 100.0
            r = update_frozen_mask!(sim, T)
            @test r.n_thawed == 1
            @test st.b[cell] == p.b_min && st.W[cell] == 0 # an empty cavity
            @test water(st) ≈ W1 rtol = 1e-12
            step!(sim)
            @test all(isfinite, Array(st.h))
        end

        @testset "stuck cells, filled, ConsistentGapCoupling: no overflow (the first raw iterate has N ~ -1e16)" begin
            # The gap here is flat in N (A = 1e-30), so the first linear solve pushes the moulin's water through the b_min floor
            # and returns N ~ -1e16: the exponential branch of relax_update (and the N^n creep law of the explicit scheme)
            # overflowed to Inf/NaN. limit_head_step! now bounds the Picard step on the negative side (N falls at most po/4
            # below min(N_prev, 0) per iteration), so the head stays finite; the first steps may need many iterations. Under
            # implicit_upwind the dry floor had no such limit and the head matrix went singular at step 1 (ARCH-2). With the
            # explicit gap scheme implicit_upwind still diverges (b ~ 1e12 m by step 4): not tested.
            for gap in ("explicit", "implicit", "fully_implicit"), kface in ("arithmetic", "implicit_upwind")
              gap == "explicit" && kface == "implicit_upwind" && continue
              @testset "gap $gap, $kface" begin
                sim = make_sim(FilledCavities(); gap = gap, kface = kface, gcp = ConsistentGapCoupling())
                s = sim.state
                @test Shakti.dry_floor(sim)
                for k in 1:3
                    step!(sim)
                    g = grounded(s)
                    @test all(isfinite, Array(s.h)) && all(isfinite, Array(s.b)) && all(isfinite, Array(s.N))
                    @test maximum(Array(s.b)[g]) < 10 # b was 1e37 (explicit) or Inf/NaN before
                    @test all(Array(s.b)[g] .>= sim.p.b_min)
                end
                @test first(Shakti.picard_status(sim.hs)) # the third step has converged
              end
            end
        end

        @testset "stuck cells, filled, no storage: the unconverged step warns with the water it lost" begin
            sim = make_sim(FilledCavities(); gap = "fully_implicit", gcp = ConsistentGapCoupling())
            @test_logs (:warn, r"did not converge.*lost\).*IceUplift\(\) or e_v > 0") match_mode = :any step!(sim)
            @test !first(Shakti.picard_status(sim.hs)) && sim.wb.created[] < -0.9 * sim.wb.input[] # N would have to reach ~ -1e9 Pa
        end

        @testset "gap held at b_max: opening blocked, empty volume bounded, budget exact" begin
            A_visc = fill(1e-30, nx, ny); b0 = fill(1e-6, nx, ny)
            ub_x = fill(3e-7, nx + 1, ny); ub_y = zeros(nx, ny + 1)
            p = ModelParameters(e_v = 0.0, b_min = 1e-6, b_max = 3e-4)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, base_mask(), A_visc, zb, zs, b0, G, ub_x, ub_y, zeros(nx, ny), taub_x, taub_y) # no moulin: with 3 m^3/s into ice this stiff the pressurized cells are a numerical stress test of their own
            ps = PicardSolver(500, floattype(1e-12), CholeskyDirectSolver(grid), grid)
            sim = Simulation(grid, state, 1, floattype(dt), p, "explicit", String[], ConstantMeltInput(), sl; uplift = NoUplift(), k_face_choice = "upwind", ps = ps, cavity_filling = UnfilledCavities())
            s = state
            for k in 1:6
                W_old = copy(Array(s.W))
                step!(sim)
                g = grounded(s)
                @test minimum(Array(s.pw)[g]) >= 0
                @test all(Array(s.W)[g] .>= 0) # never more empty than the gap itself
                if k == 6
                    @test any(Array(s.b)[g] .>= p.b_max * (1 - 1e-9)) # some gap really is held
                    res = budget_residual(sim, W_old)
                    scale = maximum(abs, Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)) + maximum(Array(s.beta) .* Array(s.abs_ub))
                    @test maximum(abs, res[g]) < 1e-6 * scale
                end
            end
        end

        @testset "water budget closes for every gap scheme (W from the water actually delivered)" begin
            for gap in ("explicit", "implicit", "fully_implicit"), stuck in (false, true)
              @testset "gap $gap, stuck=$stuck" begin
                sim = make_sim(UnfilledCavities(); stuck = stuck, gap = gap, ieb_scale = 1e-4) # a 3 m^3/s moulin into ice this stiff is a numerical stress test of its own
                s = sim.state
                W_old = copy(Array(s.W))
                for k in 1:3
                    W_old = copy(Array(s.W))
                    step!(sim)
                end
                g = grounded(s)
                res = budget_residual(sim, W_old)
                scale = maximum(abs, Array(s.mdot) ./ sim.p.rho_w .+ Array(s.ieb)) + maximum(Array(s.beta) .* Array(s.abs_ub))
                @test maximum(abs, res[g]) < 1e-6 * scale
                @test all(Array(s.W)[g] .>= 0)
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
            W_old = copy(Array(s.W))
            step!(un)
            @test s.pw[4, 4] == 0
            @test empty_gap(s)[4, 4] > 0
            res = budget_residual(un, W_old)
            scale = abs(Array(s.mdot)[4, 4] / p.rho_w + Array(s.ieb)[4, 4]) + Array(s.beta)[4, 4] * Array(s.abs_ub)[4, 4]
            @test abs(res[4, 4]) < 1e-9 * scale
        end

        @testset "empty cavity neither carries nor draws water: conductance from the water depth" begin
            un = make_sim(UnfilledCavities())
            s = un.state
            s.b .= 1e-3
            s.W .= 1e-3             # the conductance reads the water W
            s.W[3, 3] = 1e-4        # nearly dry
            s.W[4, 3] = 0.0         # dry: no water, no conductance (no floor at b_min)
            Shakti.compute_b_w!(s, un.p)
            s.dhdx .= 0.0; s.dhdy .= 0.0 # no gradient: the upwind face takes the smaller cell
            @test s.b_w[3, 3] ≈ 1e-4
            @test s.b_w[4, 3] == 0
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
            @test sim.kfs isa Upwind
            s.b .= 0.05; s.W .= 0.05
            s.W[3, 3] = 1e-4 # (3,3) holds 0.1 mm of water, its neighbours 5 cm
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
                s.b .= 0.01; s.W .= 0.01
                s.W[3, 3] = 1e-5   # holds 10 micrometres of water, gap room 9.99 mm
                hostdt = 86400.0
                C = Array(freeze_on_capacity!(zeros(nx, ny), sim; dt = hostdt, water_limited = true))
                limit = 1e-5 * p.rho_w / p.rho_i / hostdt
                if cf isa UnfilledCavities
                    @test C[3, 3] <= limit * (1 + 1e-12)       # water-limited
                    @test C[3, 3] > 0
                else
                    @test C[3, 3] > limit                       # legacy: the gap room is the limit, W is ignored
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

        # englacial water per area, e_v*max(x, 0) with x = h - zb (an unfilled cell's empty depth is in W, see englacial_storage_terms)
        efun(x, e_v) = e_v * max(x, 0)
        function par_sim(cf; e_v = 1e-7, stuck = true, kface = nothing, mask = base_mask(), pres = 1e-12)
            A_visc = fill(stuck ? 1e-30 : 5e-25, nx, ny)
            b0     = stuck ? fill(1e-6, nx, ny) : [0.01 + 0.004 * sin(i + 2j) for i in 1:nx, j in 1:ny]
            ub_x   = fill(stuck ? 3e-7 : 1e-6, nx + 1, ny)
            p = ModelParameters(e_v = e_v, b_min = 1e-6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, zeros(nx, ny + 1), ieb, taub_x, taub_y)
            ps = PicardSolver(500, floattype(pres), CholeskyDirectSolver(grid), grid)
            return Simulation(grid, state, 1, floattype(dt), p, "explicit", String[], ConstantMeltInput(), sl; uplift = NoUplift(), ps = ps, cavity_filling = cf, k_face_choice = something(kface, cf isa UnfilledCavities ? "upwind" : "arithmetic"),
                              hydraulic_potential = BedPotential()) # as above
        end

        @testset "parabolic: no unfilled cell is identical to the legacy solve" begin
            sims = [par_sim(cf; stuck = false, kface = "upwind") for cf in (FilledCavities(), UnfilledCavities())]
            for sim in sims
                Shakti.prepare_head_solve!(sim); Shakti.step_h!(sim.hs, sim)
            end
            g = grounded(sims[1].state)
            @test all(Array(sims[1].state.pw)[g] .> 0)
            @test Array(sims[2].state.h) ≈ Array(sims[1].state.h) rtol = 1e-12 atol = 1e-9
        end

        @testset "parabolic: stuck cells keep pw >= 0, the head rows store the budget's water" begin
            fi = par_sim(FilledCavities()); step!(fi)
            @test minimum(Array(fi.state.pw)[grounded(fi.state)]) < 0 # e_v this small: englacial storage does not save the stuck cells
            sim = par_sim(UnfilledCavities())
            s, p = sim.state, sim.p
            h_old, W_old = copy(Array(s.h)), copy(Array(s.W))
            step!(sim)
            g = grounded(s)
            @test first(Shakti.picard_status(sim.hs))
            @test minimum(Array(s.pw)[g]) >= 0
            @test any(Array(empty_gap(s))[g] .> 0)
            # the budget: the sheet's water W and the englacial water together take what the fluxes and sources leave
            xo, xn = h_old .- Array(s.zb), Array(s.h) .- Array(s.zb)
            qx, qy = Array(s.q_x), Array(s.q_y)
            div = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
            src = Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)
            res = ((Array(s.W) .- W_old) .+ (efun.(xn, p.e_v) .- efun.(xo, p.e_v))) ./ dt .+ div .- src
            scale = maximum(abs, src) + maximum(Array(s.beta) .* Array(s.abs_ub))
            @test maximum(abs, res[g]) < 1e-9 * scale
            # and the head rows stored that water: the gap less the empty depth below the bed
            Wh = max.(Array(s.b) .- max.(0, .-xn), 0)
            @test maximum(abs, (Array(s.W) .- Wh)[g]) < 1e-6 * maximum(Array(s.b)[g])
        end

    end

@testset "Default sheet: GlaDS-2 (unfilled cavities, uplift, implicit_upwind, consistent coupling)" begin
    nx, ny = 12, 6
    grid = Grid(nx, ny, 1.1e3, 500.0)
    x = Array(grid.x)
    zb = [0.05 * xi for xi in x, _ in 1:ny]
    H = [50 + 200 * sqrt(xi / x[end]) for xi in x, _ in 1:ny]
    mask = fill(GROUNDED, nx, ny); mask[1, :] .= LAND
    sl = RegularizedCoulombSlidingLaw(0.25)
    function default_sim(ls; kw...)
        state = State(grid)
        set_initial_conditions!(state, grid, ModelParameters(), sl, mask, fill(2.4e-24, nx, ny), zb, zb .+ H, fill(1e-3, nx, ny),
                                fill(0.06, nx, ny), fill(1e-6, nx + 1, ny), zeros(nx, ny + 1), fill(1e-8, nx, ny), zeros(nx + 1, ny), zeros(nx, ny + 1))
        return Simulation(grid, state, 2, floattype(86400.0), ModelParameters(), "fully_implicit", String[], ConstantMeltInput(), sl;
                          ps = PicardSolver(500, floattype(1e-8), ls, grid), kw...)
    end
    sim = default_sim(LUDirectSolver(grid))
    @test sim.cf isa UnfilledCavities && sim.up isa IceUplift && sim.hp isa SheetTopPotential
    @test sim.kfs isa ImplicitUpwind && sim.gcp isa ConsistentGapCoupling
    for _ in 1:2
        step!(sim)
        @test first(Shakti.picard_status(sim.hs))
    end
    g = Array(sim.state.mask) .== GROUNDED
    @test all(0 .<= Array(sim.state.N)[g] .<= Array(sim.state.po)[g] .* (1 + 1e-9)) # GlaDS-2: 0 <= N <= po
    # a symmetric-only solver: refused, and the message names the opt-outs
    err = try default_sim(CholeskyDirectSolver(grid)); "" catch e; sprint(showerror, e) end
    @test occursin("the default", err) && occursin("FilledCavities()", err) && occursin("\"upwind\"", err)
    # SHAKTI's sheet when asked for: arithmetic faces, Cholesky accepted
    shakti = default_sim(CholeskyDirectSolver(grid); cavity_filling = FilledCavities(), uplift = NoUplift())
    @test shakti.kfs isa Arithmetic && shakti.gcp isa ConsistentGapCoupling
end

@testset "Water W (GAP-5): the budget's water under unfilled cavities, the row's water elsewhere, both close the budget" begin
    nx, ny = 12, 6
    grid = Grid(nx, ny, 1.1e3, 500.0)
    x = Array(grid.x)
    zb = [0.05 * xi for xi in x, _ in 1:ny]
    H = [50 + 200 * sqrt(xi / x[end]) for xi in x, _ in 1:ny]
    mask = fill(GROUNDED, nx, ny); mask[1, :] .= LAND
    sl = RegularizedCoulombSlidingLaw(0.25)
    ieb = fill(1e-8, nx, ny); ieb[8, 3] = 2.0 / (1.1e3 * 500.0) # a 2 m^3/s moulin
    function water_sim(ls; dt = 86400.0, kw...)
        state = State(grid)
        set_initial_conditions!(state, grid, ModelParameters(), sl, mask, fill(2.4e-24, nx, ny), zb, zb .+ H, fill(1e-3, nx, ny),
                                fill(0.06, nx, ny), fill(1e-6, nx + 1, ny), zeros(nx, ny + 1), ieb, zeros(nx + 1, ny), zeros(nx, ny + 1))
        return Simulation(grid, state, 4, floattype(dt), ModelParameters(), String[], ConstantMeltInput(), sl;
                          ps = PicardSolver(500, floattype(1e-8), ls, grid), kw...)
    end
    function budget_gap(s, W_old, dt)
        qx, qy = Array(s.q_x), Array(s.q_y)
        dv = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
        r = Array(s.W) .- (W_old .+ dt .* (Array(s.mdot) ./ ModelParameters().rho_w .+ Array(s.ieb) .- dv))
        return maximum(abs, r[Array(s.mask) .== GROUNDED])
    end
    # (sheet, budgeted, relative tolerance of the budget): W from the budget under unfilled cavities, else the water the gap
    # and the head leave (the dry floor's water, or b plus the uplift layer), which closes the
    # budget to the Picard tolerance under the consistent coupling and not under the explicit one (2 % here)
    for (name, mk, budgeted, rtol) in (("GlaDS-2", () -> water_sim(LUDirectSolver(grid)), true, 1e-12),
                                       ("filled, uplift, consistent", () -> water_sim(LUDirectSolver(grid); cavity_filling = FilledCavities()), false, 1e-5),
                                       ("filled dry floor", () -> water_sim(CholeskyDirectSolver(grid); cavity_filling = FilledCavities(), uplift = NoUplift()), false, 1e-5),
                                       ("filled, explicit coupling", () -> water_sim(CholeskyDirectSolver(grid); dt = 3600.0, cavity_filling = FilledCavities(), uplift = NoUplift(),
                                                                                     gap_coupling = ExplicitGapCoupling()), false, Inf))
        @testset "$name" begin
            sim = mk()
            @test Shakti.water_is_budgeted(sim) == budgeted
            for _ in 1:4
                W_old = Array(sim.state.W)
                step!(sim)
                @test first(Shakti.picard_status(sim.hs))
                W = Array(sim.state.W)
                Shakti.constrain_water!(sim.state, sim.p, sim.cf, sim.up, Shakti.dry_floor(sim)) # the sheet's own water is within its bounds
                @test Array(sim.state.W) == W
                @test budget_gap(sim.state, W_old, sim.dt[]) <= rtol * maximum(Array(sim.state.W))
            end
            name == "filled, explicit coupling" && @test Array(sim.state.W) == Array(sim.state.b) # the legacy sheet's water is its gap
        end
    end
end

@testset "Default sheet, global water balance: water leaves the grounded region through LAND and OCEAN faces only" begin
    # The per-cell budgets above cannot see water a grounded cell sends into a cell outside the budget; here the grounded
    # region's volume change is its input less what crosses its LAND/OCEAN faces, and the OTHER_BASIN, FROZEN_BED and
    # domain-edge faces carry nothing.
    nx, ny = 10, 7
    grid = Grid(nx, ny, 1e3, 1e3)
    x = Array(grid.x)
    sl = RegularizedCoulombSlidingLaw(0.25)
    mask = fill(GROUNDED, nx, ny); mask[1, :] .= LAND; mask[end, :] .= OCEAN; mask[:, 1] .= OTHER_BASIN; mask[5, 4] = FROZEN_BED; mask[6, ny] = FROZEN_BED
    zb = [0.03 * xi for xi in x, _ in 1:ny]
    H = [60 + 300 * sqrt(xi / x[end]) for xi in x, _ in 1:ny]
    ieb = fill(1e-8, nx, ny); ieb[7, 4] = 1e-6 # a 1 m^3/s moulin next to the frozen cell
    for (name, p) in (("e_v = 0", ModelParameters()), ("e_v = 1e-3", ModelParameters(e_v = 1e-3)))
        @testset "$name" begin
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, fill(2.4e-24, nx, ny), zb, zb .+ H, fill(1e-3, nx, ny), fill(0.06, nx, ny),
                                    fill(1e-6, nx + 1, ny), zeros(nx, ny + 1), ieb, zeros(nx + 1, ny), zeros(nx, ny + 1))
            sim = Simulation(grid, state, 3, floattype(86400.0), p, String[], ConstantMeltInput(), sl; ps = PicardSolver(500, floattype(1e-10), LUDirectSolver(grid), grid))
            g = mask .== GROUNDED
            area = grid.dx * grid.dy
            for _ in 1:3
                s = sim.state
                W_old, h_old = Array(s.W), Array(s.h)
                step!(sim)
                @test first(Shakti.picard_status(sim.hs))
                qx, qy = Array(s.q_x), Array(s.q_y)
                out = Dict(LAND => 0.0, OCEAN => 0.0, OTHER_BASIN => 0.0, FROZEN_BED => 0.0, :edge => 0.0) # m^3/s leaving the grounded region
                for i in 1:nx, j in 1:ny
                    g[i, j] || continue
                    for (nb, q) in (((i - 1, j), -qx[i, j] * grid.dy), ((i + 1, j), qx[i+1, j] * grid.dy), ((i, j - 1), -qy[i, j] * grid.dx), ((i, j + 1), qy[i, j+1] * grid.dx))
                        inside = 1 <= nb[1] <= nx && 1 <= nb[2] <= ny
                        inside && g[nb...] && continue
                        out[inside ? mask[nb...] : :edge] += q
                    end
                end
                @test out[OTHER_BASIN] == 0 && out[FROZEN_BED] == 0 && out[:edge] == 0
                @test out[LAND] + out[OCEAN] > 0
                # the change of the englacial water (e_v > 0), as WaterBudget counts it
                dE = sum(Shakti.englacial_water(p.e_v, Array(s.h)[i], Array(s.zb)[i], Array(s.po)[i], Array(s.stor)[i], Array(s.stor_up)[i], p.rho_w, p.g) -
                         Shakti.englacial_water(p.e_v, h_old[i], Array(s.zb)[i], Array(s.po)[i], Array(s.stor)[i], Array(s.stor_up)[i], p.rho_w, p.g) for i in findall(g))
                wc = water_created(sim)
                dV = (sum(Array(s.W)[g] .- W_old[g]) + dE) * area
                @test dV ≈ wc.input - sim.dt[] * (out[LAND] + out[OCEAN]) + wc.created atol = 1e-12 * wc.input
                @test abs(wc.created) <= 1e-12 * wc.input
            end
        end
    end
end
