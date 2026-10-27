    @testset "LinearSlidingLaw" begin

        # Same nontrivial mask/state as the first testset above.
        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        state = State(grid)
        p = ModelParameters(e_v = 0.0)

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

        @testset "scalar C" begin
            C = 0.25
            sl = LinearSlidingLaw(grid, C)

            # A uniform scalar should stagger to a uniform Cx2/Cy2 = C^2 everywhere.
            @test all(sl.Cx2 .≈ C^2)
            @test all(sl.Cy2 .≈ C^2)

            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            # taub = C^2*N*u_b exactly -- see sliding_law.jl's LinearSlidingLaw docstring for the
            # magnitude/direction cancellation that makes this exact, not just an approximation.
            # face N: mean of the two cells, or the GROUNDED cell's own N on a face it shares with a
            # non-GROUNDED cell (sliding is kept on those faces)
            gr(i, j) = state.mask[i, j] == GROUNDED
            for j in 1:ny, i in 2:nx
                Nf = gr(i, j) == gr(i-1, j) ? (state.N[i, j] + state.N[i-1, j]) / 2 : (gr(i, j) ? state.N[i, j] : state.N[i-1, j])
                @test state.taub_x[i, j] ≈ C^2 * Nf * state.ub_x[i, j]
            end

            @test all(isfinite, Array(state.taub_x))
            @test all(isfinite, Array(state.taub_y))
        end

        @testset "spatial C" begin
            C = [0.1 + 0.01*i + 0.02*j for i in 1:nx, j in 1:ny] # non-uniform, exercises real staggering

            sl = LinearSlidingLaw(grid, C)

            # Interior x-faces average the two neighbouring cells, then square.
            for j in 1:ny, i in 2:nx
                @test sl.Cx2[i, j] ≈ ((C[i-1, j] + C[i, j]) / 2)^2
            end
            # Boundary x-faces duplicate the nearest cell.
            @test all(sl.Cx2[1, :]    .≈ C[1, :]  .^ 2)
            @test all(sl.Cx2[nx+1, :] .≈ C[nx, :] .^ 2)

            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            @test all(isfinite, Array(state.taub_x))
            @test all(isfinite, Array(state.taub_y))
        end

        @testset "Picard solve converges" begin
            sl = LinearSlidingLaw(grid, 0.25)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

            ls = CholeskyDirectSolver(grid)
            ps = PicardSolver(500, 1e-6, ls, grid)
            mt = MeltTerms{p.mdot_includes_G, p.mdot_includes_frictional, p.mdot_includes_potential, p.mdot_includes_sensible, p.mdot_includes_qT}()
            elliptic_solver!(ps, state, grid, p, mt, Arithmetic(), sl)

            @test ps.converged
            @test all(isfinite, Array(state.h))
            @test all(isfinite, Array(state.N))
        end

        @testset "N < 0: no shear, no frictional heat" begin
            # A face with water above overburden carries no basal shear under any law; before, the stress pointed along
            # the sliding direction with size C|N| and the frictional heat (abs) counted it as melt.
            ux = fill(1e-6, nx + 1, ny); uy = fill(5e-7, nx, ny + 1)
            laws = (RegularizedCoulombSlidingLaw(0.25), RegularizedCoulombFieldSlidingLaw(grid, 0.25),
                    RegularizedCoulombV0SlidingLaw(grid, 0.25, 1e-5), LinearSlidingLaw(grid, 0.25),
                    BuddSlidingLaw(grid, 1e-3), CoulombSlidingLaw(grid, 0.5), TsaiSlidingLaw(grid, 1e4, 0.5),
                    SchoofSlidingLaw(grid, 1e3, 0.5), GagliardiniSlidingLaw(grid, 0.5, 1e-20; q = 2),
                    HelanowSlidingLaw(grid, 0.5, 1e-4), PseudoPlasticSlidingLaw(grid, 0.5),
                    LinearCombinationSlidingLaw(grid, 1e3, 0.5, 1e12))
            for sl in laws
                s = State(grid)
                set_initial_conditions!(s, grid, p, sl, mask, A_visc, zb, zs, b, G, ux, uy, ieb, taub_x, taub_y)
                N0 = Array(s.N)
                @test all(>(0), N0[mask .== GROUNDED])
                s.N[2:4, 2:4] .= -1e5                   # a patch above overburden
                compute_taub_xy!(s, p, sl)
                @test s.taub_x[3, 3] == 0 && s.taub_x[4, 3] == 0 && s.taub_y[3, 3] == 0 && s.taub_y[3, 4] == 0
                compute_mdot!(s, p, MeltTerms{false, true, false, false, false}()) # writes the frictional heat Q_b
                @test s.Q_b[3, 3] == 0
                @test s.Q_b[5, 5] > 0                   # outside the patch: unchanged
                @test all(>=(0), Array(s.taub_x) .* Array(s.ub_x)) && all(>=(0), Array(s.taub_y) .* Array(s.ub_y))
            end
        end

        @testset "literature laws match their formulas" begin
            # Each law's taub on the interior faces between two GROUNDED cells, against its formula written out here
            # with the face N and |u_b| the means of the two cells.
            ux = fill(1e-6, nx + 1, ny); uy = fill(5e-7, nx, ny + 1)
            yr = 31_557_600
            Ccell = [0.4 + 0.02*i + 0.01*j for i in 1:nx, j in 1:ny]
            laws = [
                (WeertmanSlidingLaw(grid, 2e4; m = 1/3),                (N, v, C) -> 2e4 * v^(1/3)),
                (WeertmanSlidingLaw(grid, 3e9; m = 1),                  (N, v, C) -> 3e9 * v),
                (BuddSlidingLaw(grid, 1e-2; m = 1/3, q = 1),            (N, v, C) -> 1e-2 * N * v^(1/3)),
                (CoulombSlidingLaw(grid, Ccell),                        (N, v, C) -> C * N),
                (TsaiSlidingLaw(grid, 1e4, 0.5; m = 1/3),               (N, v, C) -> min(1e4 * v^(1/3), 0.5N)),
                (SchoofSlidingLaw(grid, 1e3, 0.5; m = 1/3),             (N, v, C) -> 1e6 * v^(1/3) / (1 + (1e6 / (0.5N))^3 * v)^(1/3)),
                (GagliardiniSlidingLaw(grid, 0.5, 1e-20; q = 2, n = 3), (N, v, C) -> (chi = v / ((0.5N)^3 * 1e-20); 0.5N * (chi / (1 + chi^2 / 4))^(1/3))),
                (HelanowSlidingLaw(grid, Ccell, 1e-4; m = 3),           (N, v, C) -> C * N * (v / (v + (1e-4N)^3))^(1/3)),
                (PseudoPlasticSlidingLaw(grid, 0.5; q = 0.25, c0 = 1e3), (N, v, C) -> (1e3 + 0.5N) * (v * yr / 100)^0.25),
                (LinearCombinationSlidingLaw(grid, 1e3, 0.5, 1e12),     (N, v, C) -> (xi = 1e12v / N^3; (0.5N * xi + 1e6 * v^(1/3)) / (1 + xi))),
                (FunctionSlidingLaw(grid, (N, v, C) -> C * N * v^0.2, Ccell), (N, v, C) -> C * N * v^0.2),
            ]
            gr(s, i, j) = s.mask[i, j] == GROUNDED
            for (sl, T) in laws
                s = State(grid)
                set_initial_conditions!(s, grid, p, sl, mask, A_visc, zb, zs, b, G, ux, uy, ieb, taub_x, taub_y)
                compute_taub_xy!(s, p, sl)
                for j in 1:ny, i in 2:nx
                    gr(s, i, j) && gr(s, i-1, j) || continue
                    N = (s.N[i, j] + s.N[i-1, j]) / 2
                    v = (s.abs_ub[i, j] + s.abs_ub[i-1, j]) / 2
                    C = (Ccell[i, j] + Ccell[i-1, j]) / 2
                    @test s.taub_x[i, j] ≈ T(N, v, C) * s.ub_x[i, j] / v
                end
                for j in 2:ny, i in 1:nx
                    gr(s, i, j) && gr(s, i, j-1) || continue
                    N = (s.N[i, j] + s.N[i, j-1]) / 2
                    v = (s.abs_ub[i, j] + s.abs_ub[i, j-1]) / 2
                    C = (Ccell[i, j] + Ccell[i, j-1]) / 2
                    @test s.taub_y[i, j] ≈ T(N, v, C) * s.ub_y[i, j] / v
                end
                @test all(isfinite, Array(s.taub_x)) && all(isfinite, Array(s.taub_y))
            end
        end

        @testset "regelation + creep solves its implicit law" begin
            ux = [1e-6 * (1 + 0.5sin(i + j)) for i in 1:nx+1, j in 1:ny]; uy = fill(5e-7, nx, ny + 1)
            for (A_R, A_C) in ((1e-11, 1e-22), (1e-14, 1e-20), (1e-9, 1e-25)) # both mechanisms, creep-, regelation-dominated
                sl = RegelationCreepSlidingLaw(grid, A_R, A_C; n = 3)
                s = State(grid)
                set_initial_conditions!(s, grid, p, sl, mask, A_visc, zb, zs, b, G, ux, uy, ieb, taub_x, taub_y)
                for j in 1:ny, i in 2:nx
                    gr = s.mask[i, j] == GROUNDED, s.mask[i-1, j] == GROUNDED
                    gr[1] == gr[2] || continue
                    v = (s.abs_ub[i, j] + s.abs_ub[i-1, j]) / 2
                    tau = s.taub_x[i, j] * v / s.ub_x[i, j]
                    @test tau > 0
                    @test A_R * tau + A_C * tau^3 ≈ v rtol = 1e-12
                end
            end
        end

        @testset "limits and equivalences" begin
            g1 = grid # a single face is enough: the laws are evaluated pointwise
            stress(sl, N, v) = Shakti.basal_stress(Shakti.kernel_args(sl, p)[1], N, v, 1.0, map(C -> C[1, 1], Shakti.kernel_args(sl, p)[2]))
            N = 2e5
            # Schoof: Weertman C^2 v^m at low speed, Cmax N at high speed
            sch = SchoofSlidingLaw(g1, 1e3, 0.5)
            @test stress(sch, N, 1e-14) ≈ 1e6 * (1e-14)^(1/3) rtol = 1e-3
            @test stress(sch, N, 1e3) ≈ 0.5N rtol = 1e-3
            # Cornford et al. (2020)'s form is Schoof's with C -> sqrt(C), Cmax = alpha^2
            Cc, a2, v = 1e6, 0.5, 3e-6
            @test stress(sch, N, v) ≈ Cc * a2 * N * v^(1/3) / (Cc^3 * v + (a2 * N)^3)^(1/3)
            # Tsai: the smaller branch
            tsai = TsaiSlidingLaw(g1, 1e4, 0.5)
            @test stress(tsai, N, 1e-9) ≈ 1e4 * 1e-3
            @test stress(tsai, N, 1e6) ≈ 0.5N
            # Gagliardini at q = 1 is Helanow's with K^m = As C^n
            As, C = 1e-20, 0.5
            @test stress(GagliardiniSlidingLaw(g1, C, As; q = 1), N, v) ≈ stress(HelanowSlidingLaw(g1, C, (As * C^3)^(1/3); m = 3), N, v)
            # Gagliardini with q > 1 peaks at C N and softens beyond (rate weakening)
            gag = GagliardiniSlidingLaw(g1, C, As; q = 2)
            vs = 10 .^ range(-12, 2; length = 400)
            T = [stress(gag, N, v) for v in vs]
            @test maximum(T) <= C * N * (1 + 1e-12) && maximum(T) > 0.99C * N
            @test T[end] < 0.5maximum(T)
            # linear combination: Weertman at low speed, Coulomb at high
            lc = LinearCombinationSlidingLaw(g1, 1e3, 0.5, 1e12)
            @test stress(lc, N, 1e-16) ≈ 1e6 * (1e-16)^(1/3) rtol = 1e-3
            @test stress(LinearCombinationSlidingLaw(g1, 1e3, 0.5, 1e18), N, 1e6) ≈ 0.5N rtol = 1e-3
            # pseudo-plastic: tau_c at the threshold speed, Coulomb at q = 0
            @test stress(PseudoPlasticSlidingLaw(g1, 0.5; u0 = 1e-5), N, 1e-5) ≈ 0.5N
            @test stress(PseudoPlasticSlidingLaw(g1, 0.5; q = 0), N, 3e-7) ≈ stress(CoulombSlidingLaw(g1, 0.5), N, 3e-7)
            # the cavitation laws carry no shear at N = 0
            for sl in (sch, gag, lc, HelanowSlidingLaw(g1, C, 1e-4), CoulombSlidingLaw(g1, 0.5), TsaiSlidingLaw(g1, 1e4, 0.5))
                @test stress(sl, 0.0, v) == 0
            end
        end

        @testset "Picard solve converges with every law" begin
            ls = CholeskyDirectSolver(grid)
            mt = MeltTerms{p.mdot_includes_G, p.mdot_includes_frictional, p.mdot_includes_potential, p.mdot_includes_sensible, p.mdot_includes_qT}()
            for sl in (WeertmanSlidingLaw(grid, 2e4), BuddSlidingLaw(grid, 1e-2), CoulombSlidingLaw(grid, 0.5),
                       TsaiSlidingLaw(grid, 1e4, 0.5), SchoofSlidingLaw(grid, 1e3, 0.5), GagliardiniSlidingLaw(grid, 0.5, 1e-20; q = 2),
                       HelanowSlidingLaw(grid, 0.5, 1e-4), PseudoPlasticSlidingLaw(grid, 0.5), RegelationCreepSlidingLaw(grid, 1e-11, 1e-22),
                       LinearCombinationSlidingLaw(grid, 1e3, 0.5, 1e12), FunctionSlidingLaw(grid, (N, v, C) -> C * N, 0.5))
                s = State(grid)
                set_initial_conditions!(s, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
                ps = PicardSolver(500, 1e-6, ls, grid)
                elliptic_solver!(ps, s, grid, p, mt, Arithmetic(), sl)
                @test ps.converged
                @test all(isfinite, Array(s.taub_x)) && all(isfinite, Array(s.N))
            end
        end

    end
