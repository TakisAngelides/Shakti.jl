    @testset "FROZEN_BED" begin

        # Same nontrivial mask/state fixture as the other test files (GROUNDED interior,
        # OCEAN/LAND/OTHER_BASIN edges), plus an interior cell available for FROZEN_BED.
        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        p = ModelParameters(e_v = 0.0, p_atm = 1000.0, b_min = 1e-3) # nonzero p_atm distinguishes FROZEN_BED's pw:=0 convention from LAND/OTHER_BASIN's pw:=p_atm; nonzero b_min distinguishes thaw's reseed from freeze's b:=0
        sl = RegularizedCoulombSlidingLaw(0.25)

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

        base_mask() = begin
            mask = fill(GROUNDED, nx, ny)
            mask[end, :] .= OCEAN
            mask[1, :]   .= LAND
            mask[:, 1]   .= OTHER_BASIN
            mask
        end

        @testset "set_initial_conditions! with a FROZEN_BED cell already in the mask" begin
            mask = base_mask()
            mask[3, 3] = FROZEN_BED # otherwise-GROUNDED interior cell

            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            @test state.mask[3, 3] == FROZEN_BED
            @test state.b[3, 3] == 0.0
            @test state.pw[3, 3] == 0.0 # exactly 0, NOT p.p_atm -- catches a regression that folds FROZEN_BED into LAND/OTHER_BASIN's branch
            @test state.po[3, 3] > 0 # sanity: not a vacuous 0 ≈ 0 check below
            @test state.N[3, 3] ≈ state.po[3, 3] # pw=0 => N = po - 0 = po exactly

            # Faces touching the frozen cell are invalid, same convention as OTHER_BASIN.
            @test state.valid_x[3, 3] == 0.0
            @test state.valid_x[4, 3] == 0.0
            @test state.valid_y[3, 3] == 0.0
            @test state.valid_y[3, 4] == 0.0
            # An ordinary GROUNDED/GROUNDED face elsewhere stays valid.
            @test state.valid_x[3, 4] == 1.0
        end

        @testset "freeze_cells!/thaw_cells! round-trip" begin
            mask = base_mask()
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            # Give the target cell a real, nonzero pw first -- freezing it is then a genuine test
            # of resetting pw, not a no-op on an already-zero field.
            state.pw[3, 3] = 5e4
            compute_N!(state, p)
            @test state.pw[3, 3] != 0.0

            freeze_mask = falses(nx, ny)
            freeze_mask[3, 3] = true
            freeze_mask[end, 3] = true # also mark an OCEAN cell -- must be a no-op (only currently-GROUNDED cells freeze)

            freeze_cells!(state, p, freeze_mask)

            @test state.mask[3, 3] == FROZEN_BED
            @test state.mask[end, 3] == OCEAN # untouched
            @test state.b[3, 3] == 0.0
            @test state.lc[3, 3] == 0.0 # lc follows b
            @test state.pw[3, 3] == 0.0
            @test state.h[3, 3] ≈ state.zb[3, 3] # h = pw/(rho_w*g) + zb, pw=0 => h=zb
            @test state.K[3, 3] == 0.0
            @test state.N[3, 3] ≈ state.po[3, 3]
            @test state.valid_x[3, 3] == 0.0

            # An untouched interior GROUNDED cell is unaffected.
            @test state.mask[4, 4] == GROUNDED
            @test state.b[4, 4] == 0.01

            thaw_mask = falses(nx, ny)
            thaw_mask[3, 3] = true
            thaw_mask[1, 3] = true # also mark a LAND cell -- must be a no-op (only currently-FROZEN_BED cells thaw)

            thaw_cells!(state, p, thaw_mask)

            @test state.mask[3, 3] == GROUNDED
            @test state.mask[1, 3] == LAND # untouched
            @test state.b[3, 3] == p.b_min # reseeded, not left at the frozen 0.0
            @test state.lc[3, 3] == p.b_min # lagged creep length refreshed with b (StandardCreep: lc = b), not left at the frozen 0.0
            @test state.pw[3, 3] == 0.0 # thaw does NOT touch pw -- no water magically appears
            @test state.valid_x[3, 3] == 1.0
        end

        @testset "Picard solve runs with a FROZEN_BED cell present and keeps it pinned" begin
            mask = base_mask()
            mask[3, 3] = FROZEN_BED

            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            h_before = state.h[3, 3]

            ls = CholeskyDirectSolver(grid)
            ps = PicardSolver(500, 1e-6, ls, grid)
            mt = MeltTerms{p.mdot_includes_G, p.mdot_includes_frictional, p.mdot_includes_potential, p.mdot_includes_sensible, p.mdot_includes_qT}()
            elliptic_solver!(ps, state, grid, p, mt, Arithmetic(), sl)

            @test ps.converged
            @test all(isfinite, Array(state.h))
            @test all(isfinite, Array(state.N))
            @test state.h[3, 3] == h_before # frozen row: h held exactly fixed through the solve
            @test state.N[3, 3] ≈ state.po[3, 3] # still reads as full overburden after solving
        end

        T_cold_again(T) = (T2 = copy(T); T2[3, 3] = -2.0; T2)

        @testset "update_frozen_mask!: threshold, hysteresis, discarded water" begin
            pt = ModelParameters(e_v = 0.0, p_atm = 1000.0, b_min = 1e-3, T_freeze = -1.0, T_hysteresis = 0.5) # thaw threshold = -0.5
            mask = base_mask()
            state = State(grid)
            set_initial_conditions!(state, grid, pt, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            T = zeros(nx, ny)
            T[3, 3]   = -2.0 # GROUNDED, well below T_freeze -> freezes
            T[end, 3] = -2.0 # OCEAN, also cold -> must be untouched and excluded from the counts
            T[3, 4]   = -0.7 # GROUNDED, inside the hysteresis band -> must stay GROUNDED

            r = update_frozen_mask!(state, pt, T)
            @test r.n_frozen == 1
            @test r.n_thawed == 0
            @test r.discarded_b ≈ 0.01 # only the freezing cell's b, read before freeze_cells! zeroes it
            @test state.mask[3, 3] == FROZEN_BED
            @test state.b[3, 3] == 0.0
            @test state.mask[end, 3] == OCEAN
            @test state.mask[3, 4] == GROUNDED

            # Inside the band the frozen cell stays frozen (no flicker) and nothing else changes.
            T[3, 3] = -0.7
            r = update_frozen_mask!(state, pt, T)
            @test (r.n_frozen, r.n_thawed) == (0, 0)
            @test state.mask[3, 3] == FROZEN_BED

            # Above T_freeze + T_hysteresis it thaws, reseeded at b_min like thaw_cells!.
            T[3, 3] = -0.4
            r = update_frozen_mask!(state, pt, T)
            @test (r.n_frozen, r.n_thawed) == (0, 1)
            @test state.mask[3, 3] == GROUNDED
            @test state.b[3, 3] == pt.b_min

            @test_throws ArgumentError ModelParameters(T_hysteresis = -0.1)

            # b_min = 0 is allowed (no guard): thawed cells simply restart at b = 0, whether a lone cell or a block.
            p0 = ModelParameters(e_v = 0.0, p_atm = 1000.0, b_min = 0.0)
            update_frozen_mask!(state, pt, T_cold_again(T)) # (3,3) freezes again
            r = update_frozen_mask!(state, p0, zeros(nx, ny))
            @test r.n_thawed == 1
            @test state.mask[3, 3] == GROUNDED
            @test state.b[3, 3] == 0.0

            block = falses(nx, ny); block[2:4, 3:5] .= true
            freeze_cells!(state, p0, block)
            @test update_frozen_mask!(state, p0, zeros(nx, ny)).n_thawed == 9
            @test all(==(0.0), Array(state.b)[2:4, 3:5])
        end

    
        @testset "set_mask!: follow a host mask" begin
            mask = base_mask()
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            state.mask[4, 4] = FROZEN_BED                 # frozen cell the host still has grounded
            new = copy(base_mask())
            new[3, 3] = OTHER_BASIN                       # host: no longer grounded (e.g. too thin to solve)
            new[end, 3] = GROUNDED                        # host: newly grounded (was OCEAN)
            r = set_mask!(state, p, new)
            @test r.n_off == 1 && r.n_on == 1
            @test r.discarded_b ≈ b[3, 3]
            @test state.mask[3, 3] == OTHER_BASIN && state.b[3, 3] == 0
            @test state.pw[3, 3] == p.p_atm
            @test state.mask[end, 3] == GROUNDED && state.b[end, 3] == p.b_min
            @test state.mask[4, 4] == FROZEN_BED          # stays frozen
            @test state.valid_x[3, 3] == 0.0 && state.valid_x[4, 3] == 0.0   # faces of the now inert cell
            @test state.pw[end, 3] ≈ p.p_atm - p.rho_sw * p.g * min(state.zb[end, 3], 0.0)  # old OCEAN pw as the initial guess
            @test set_mask!(state, p, new).n_on == 0       # idempotent
        end

        @testset "freeze_isolated!: a GROUNDED cell cut off by frozen neighbours" begin
            mask = base_mask()
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            T_prime_b = zeros(nx, ny)
            for (i, j) in ((2, 3), (4, 3), (3, 2), (3, 4))
                T_prime_b[i, j] = -5.0                     # freeze the four neighbours of (3, 3)
            end
            r = update_frozen_mask!(state, p, T_prime_b)
            @test r.n_frozen == 4 && r.n_isolated == 1
            @test state.mask[3, 3] == FROZEN_BED && state.b[3, 3] == 0
            @test freeze_isolated!(state, p) == 0          # nothing left to isolate
            @test state.mask[2, 2] == GROUNDED             # a corner neighbour keeps its other drainage
        end

        @testset "gap_budget_terms: the budget where b is clamped" begin
            q = ModelParameters(b_max = 1.0, b_min = 1e-6)
            A, h = 5e-25, 100.0
            args(b, mdot, N) = (b, q.b_min, q.b_max, mdot, 0.0, 0.0, A, N, b, h, q.rho_w, q.rho_i, q.g, q.n, q.n_minus_1_exp)
            legacy(b, mdot, N) = Shakti.gap_budget_terms(0, args(b, mdot, N)...)
            # at b_max with opening winning: held under 1 and 2, the melt enters as water
            src0, gd0 = legacy(1.0, 1e-3, 1e5)
            @test gd0 > 0 && src0 ≈ 1e-3 * (1 / q.rho_w - 1 / q.rho_i) + A * 1e5^3 * 1.0 + gd0 * h
            @test Shakti.gap_budget_terms(1, args(1.0, 1e-3, 1e5)...) == (1e-3 / q.rho_w, 0.0)
            @test Shakti.gap_budget_terms(2, args(1.0, 1e-3, 1e5)...) == (1e-3 / q.rho_w, 0.0)
            # below the cap nothing changes
            @test Shakti.gap_budget_terms(1, args(0.5, 1e-3, 1e5)...) == legacy(0.5, 1e-3, 1e5)
            # at b_min with closure winning: held only under 2
            @test Shakti.gap_budget_terms(1, args(1e-6, 0.0, 1e6)...) == legacy(1e-6, 0.0, 1e6)
            @test Shakti.gap_budget_terms(2, args(1e-6, 0.0, 1e6)...) == (0.0, 0.0)
        end

        @testset "outflow_only: LAND/OCEAN faces do not feed the ice" begin
            mask = base_mask()                             # LAND at i = 1, OCEAN at i = nx
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            state.h[1, 3] = state.h[2, 3] + 50.0           # land head above its grounded neighbour: inflow face
            state.h[1, 4] = state.h[2, 4] - 50.0           # land head below: outflow face
            compute_face_masks!(state)                     # legacy: both open
            @test state.valid_x[2, 3] == 1 && state.valid_x[2, 4] == 1
            compute_face_masks!(state, true)
            @test state.valid_x[2, 3] == 0                 # closed: would carry land water into the ice
            @test state.valid_x[2, 4] == 1                 # open: the ice drains onto land
            @test state.valid_x[3, 3] == 1                 # GROUNDED-GROUNDED faces untouched
        end

end
