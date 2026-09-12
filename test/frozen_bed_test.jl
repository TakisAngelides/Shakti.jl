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

    end
