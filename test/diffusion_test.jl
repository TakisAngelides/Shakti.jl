    @testset "compute_D! (SUHMO Eq. 10 diffusion coefficient)" begin

        nx, ny = 4, 4
        grid = Grid(nx, ny, 1e3, 1e3)
        p = ModelParameters(e_v = 0.0)

        # Seed the face fields compute_D! reads directly with simple, known values -- bypassing
        # the full elliptic solve entirely, since compute_D! only ever reads b_x/b_y/q_x/q_y/
        # dhdx/dhdy/dpwdx/dpwdy (already face-centered, no averaging of its own), so this is a
        # faithful, self-contained test of its own formula in isolation. Deliberately set uniformly
        # nonzero even at the domain-boundary faces (unlike the real pipeline, where dhdx/dpwdx are
        # left exactly zero there by compute_dhdx_kernel!/compute_dpwdx_kernel!) so the boundary
        # check below isolates compute_D_kernel!'s OWN restricted range, not an upstream zero
        # cascading in from dhdx.
        state = State(grid)
        state.b_x   .= 0.02
        state.q_x   .= -1e-5
        state.dhdx  .= 0.1
        state.dpwdx .= 50.0
        state.b_y   .= 0.03
        state.q_y   .= -2e-5
        state.dhdy  .= 0.2
        state.dpwdy .= 80.0

        mt_full = MeltTerms{true, true, true, true, true}()

        @testset "NoDiffusion skips the update entirely" begin
            s = State(grid)
            s.b_x .= state.b_x; s.q_x .= state.q_x; s.dhdx .= state.dhdx; s.dpwdx .= state.dpwdx
            s.b_y .= state.b_y; s.q_y .= state.q_y; s.dhdy .= state.dhdy; s.dpwdy .= state.dpwdy
            compute_D!(s, p, mt_full, NoDiffusion())
            @test all(iszero, Array(s.D_x))
            @test all(iszero, Array(s.D_y))
        end

        @testset "WithDiffusion matches Eq. 10 directly, both terms on" begin
            s = State(grid)
            s.b_x .= state.b_x; s.q_x .= state.q_x; s.dhdx .= state.dhdx; s.dpwdx .= state.dpwdx
            s.b_y .= state.b_y; s.q_y .= state.q_y; s.dhdy .= state.dhdy; s.dpwdy .= state.dpwdy
            compute_D!(s, p, mt_full, WithDiffusion(CholeskyDirectSolver(grid)))

            expected_Dx = (s.b_x[2, 1] / p.rho_i) * (1 / p.L) *
                          (-p.rho_w * p.g * s.q_x[2, 1] * s.dhdx[2, 1] + p.ct * p.cw * p.rho_w * s.q_x[2, 1] * s.dpwdx[2, 1])
            @test s.D_x[2, 1] ≈ expected_Dx

            expected_Dy = (s.b_y[1, 2] / p.rho_i) * (1 / p.L) *
                          (-p.rho_w * p.g * s.q_y[1, 2] * s.dhdy[1, 2] + p.ct * p.cw * p.rho_w * s.q_y[1, 2] * s.dpwdy[1, 2])
            @test s.D_y[1, 2] ≈ expected_Dy

            # Domain-boundary faces are left exactly zero, regardless of the (deliberately nonzero)
            # q_x/dhdx values seeded there -- compute_D_kernel!'s own restricted range, not an
            # upstream zero cascading in.
            @test all(iszero, Array(s.D_x[1, :]))
            @test all(iszero, Array(s.D_x[end, :]))
            @test all(iszero, Array(s.D_y[:, 1]))
            @test all(iszero, Array(s.D_y[:, end]))
        end

        @testset "Potential/Sensible gating matches compute_mdot!'s own convention" begin
            # Potential only (Sensible off): D_x should equal just the -rho_w*g*q*dhdx term.
            s1 = State(grid)
            s1.b_x .= state.b_x; s1.q_x .= state.q_x; s1.dhdx .= state.dhdx; s1.dpwdx .= state.dpwdx
            mt_pot_only = MeltTerms{true, true, true, false, true}()
            compute_D!(s1, p, mt_pot_only, WithDiffusion(CholeskyDirectSolver(grid)))
            expected_pot_only = (s1.b_x[2, 1] / p.rho_i) * (1 / p.L) * (-p.rho_w * p.g * s1.q_x[2, 1] * s1.dhdx[2, 1])
            @test s1.D_x[2, 1] ≈ expected_pot_only

            # Sensible only (Potential off): D_x should equal just the ct*cw*rho_w*q*dpwdx term.
            s2 = State(grid)
            s2.b_x .= state.b_x; s2.q_x .= state.q_x; s2.dhdx .= state.dhdx; s2.dpwdx .= state.dpwdx
            mt_sens_only = MeltTerms{true, true, false, true, true}()
            compute_D!(s2, p, mt_sens_only, WithDiffusion(CholeskyDirectSolver(grid)))
            expected_sens_only = (s2.b_x[2, 1] / p.rho_i) * (1 / p.L) * (p.ct * p.cw * p.rho_w * s2.q_x[2, 1] * s2.dpwdx[2, 1])
            @test s2.D_x[2, 1] ≈ expected_sens_only

            # ct/cw left nonzero (ModelParameters' own default) but Sensible off should NOT leak
            # the sensible contribution back in -- the exact concern that motivated gating D by
            # mt's flags rather than by ct/cw's literal values.
            @test p.ct != 0 && p.cw != 0 # sanity: this test is only meaningful if ct/cw are actually nonzero
            @test s1.D_x[2, 1] ≈ expected_pot_only # unchanged from above: no sensible leakage
        end

        @testset "D through the real Picard loop (WithDiffusion)" begin
            sl = RegularizedCoulombSlidingLaw(0.25)
            mask = fill(GROUNDED, nx, ny)
            mask[end, :] .= OCEAN
            mask[1, :]   .= LAND
            mask[:, 1]   .= OTHER_BASIN
            A_visc = fill(5e-25, nx, ny)
            zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
            zs     = zb .+ 500.0
            b0     = fill(0.01, nx, ny)
            G      = fill(0.06, nx, ny)
            ub_x   = fill(1e-6, nx + 1, ny)
            ub_y   = zeros(nx, ny + 1)
            ieb    = zeros(nx, ny)
            ieb[3, 3] = 3 / (grid.dx * grid.dy)
            taub_x = zeros(nx + 1, ny)
            taub_y = zeros(nx, ny + 1)

            s = State(grid)
            p = ModelParameters(e_v = 0.0)
            set_initial_conditions!(s, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            @test all(iszero, Array(s.D_x)) # untouched before any WithDiffusion Picard iteration ever runs

            ls_h = CholeskyDirectSolver(grid)
            ps = PicardSolver(50, floattype(1e-6), ls_h, grid)
            ls_b = CholeskyDirectSolver(grid)
            mt = MeltTerms{true, true, true, true, true}()

            elliptic_solver!(ps, s, grid, p, mt, Arithmetic(), sl; ds = WithDiffusion(ls_b))

            @test ps.converged
            @test all(isfinite, Array(s.D_x))
            @test all(isfinite, Array(s.D_y))
            @test all(iszero, Array(s.D_x[1, :])) # boundary faces still exactly zero after a real solve
            @test all(iszero, Array(s.D_x[end, :]))
        end

    end
