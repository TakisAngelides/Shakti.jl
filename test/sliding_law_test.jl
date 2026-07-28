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
            for j in 1:ny, i in 2:nx
                Nf = (state.N[i, j] + state.N[i-1, j]) / 2
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
            shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
            elliptic_solver!(ps, state, grid, p, shs, Arithmetic(), sl)

            @test ps.converged
            @test all(isfinite, Array(state.h))
            @test all(isfinite, Array(state.N))
        end

    end
