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
                    RegularizedCoulombV0SlidingLaw(grid, 0.25, 1e-5), LinearSlidingLaw(grid, 0.25))
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

    end
