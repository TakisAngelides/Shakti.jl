    @testset "compute_D! (SUHMO Eq. 10 diffusion coefficient)" begin

        nx, ny = 4, 4
        grid = Grid(nx, ny, 1e3, 1e3)
        p = ModelParameters(e_v = 0.0)

        # Seed the fields compute_D! reads directly with simple, known values -- bypassing
        # the full elliptic solve entirely, since compute_D! only ever reads mask/b/W (cell-centred)
        # and q_x/q_y/dhdx/dhdy/dpwdx/dpwdy (face-centred, summed onto cells as mdot does), so this is a
        # faithful, self-contained test of its own formula in isolation. Deliberately set uniformly
        # nonzero even at the domain-boundary faces (unlike the real pipeline, where dhdx/dpwdx are
        # left exactly zero there by compute_dhdx_kernel!/compute_dpwdx_kernel!) so the boundary
        # check below isolates compute_D_kernel!'s OWN restricted range, not an upstream zero
        # cascading in from dhdx.
        state = State(grid)
        state.b     .= 0.02 # uniform, so the face mean of the filled gap is 0.02 on every face
        state.q_x   .= -1e-5
        state.dhdx  .= 0.1
        state.dpwdx .= 50.0
        state.q_y   .= -2e-5
        state.dhdy  .= 0.2
        state.dpwdy .= 80.0

        mt_full = MeltTerms{true, true, true, true, true}()
        # SUHMO Eq. 10 in a cell of these uniform fields: the face-to-cell sum (half of each of the four faces)
        # gives q.grad(h) = q_x*dhdx + q_y*dhdy, and likewise for pw.
        heat_pot  = p.rho_w * p.g * abs(-1e-5 * 0.1 + -2e-5 * 0.2)
        heat_sens = p.ct * p.cw * p.rho_w * (-1e-5 * 50.0 + -2e-5 * 80.0)
        D_eq10(b, heat) = max(0.0, b / (p.rho_i * p.L) * heat)

        @testset "NoDiffusion skips the update entirely" begin
            s = State(grid)
            s.b .= state.b; s.q_x .= state.q_x; s.dhdx .= state.dhdx; s.dpwdx .= state.dpwdx
            s.q_y .= state.q_y; s.dhdy .= state.dhdy; s.dpwdy .= state.dpwdy
            copyto!(s.W, s.b); compute_D!(s, p, mt_full, NoDiffusion())
            @test all(iszero, Array(s.D_x))
            @test all(iszero, Array(s.D_y))
        end

        @testset "WithDiffusion matches Eq. 10 directly, both terms on" begin
            s = State(grid)
            s.b .= state.b; s.q_x .= state.q_x; s.dhdx .= state.dhdx; s.dpwdx .= state.dpwdx
            s.q_y .= state.q_y; s.dhdy .= state.dhdy; s.dpwdy .= state.dpwdy
            copyto!(s.W, s.b); compute_D!(s, p, mt_full, WithDiffusion(CholeskyDirectSolver(grid)))

            expected = D_eq10(0.02, heat_pot + heat_sens)
            @test expected > 0
            @test s.D_x[2, 1] ≈ expected
            @test s.D_y[1, 2] ≈ expected # one scalar per cell: the same on x- and y-faces

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
            s1.b .= state.b; s1.q_x .= state.q_x; s1.dhdx .= state.dhdx; s1.dpwdx .= state.dpwdx
            mt_pot_only = MeltTerms{true, true, true, false, true}()
            s1.q_y .= state.q_y; s1.dhdy .= state.dhdy; s1.dpwdy .= state.dpwdy
            copyto!(s1.W, s1.b); compute_D!(s1, p, mt_pot_only, WithDiffusion(CholeskyDirectSolver(grid)))
            expected_pot_only = D_eq10(0.02, heat_pot)
            @test s1.D_x[2, 1] ≈ expected_pot_only

            # Sensible only (Potential off): D_x should equal just the ct*cw*rho_w*q*dpwdx term.
            s2 = State(grid)
            s2.b .= state.b; s2.q_x .= state.q_x; s2.dhdx .= state.dhdx; s2.dpwdx .= state.dpwdx
            s2.q_y .= state.q_y; s2.dhdy .= state.dhdy; s2.dpwdy .= state.dpwdy
            mt_sens_only = MeltTerms{true, true, false, true, true}()
            copyto!(s2.W, s2.b); compute_D!(s2, p, mt_sens_only, WithDiffusion(CholeskyDirectSolver(grid)))
            expected_sens_only = D_eq10(0.02, heat_sens) # D is floored at 0 (a net heat deficit is freeze-on, handled by mdot, not backward diffusion)
            @test s2.D_x[2, 1] ≈ expected_sens_only
            @test all(>=(0), Array(s2.D_x)) && all(>=(0), Array(s2.D_y))

            # ct/cw left nonzero (ModelParameters' own default) but Sensible off should NOT leak
            # the sensible contribution back in -- the exact concern that motivated gating D by
            # mt's flags rather than by ct/cw's literal values.
            @test p.ct != 0 && p.cw != 0 # sanity: this test is only meaningful if ct/cw are actually nonzero
            @test s1.D_x[2, 1] ≈ expected_pot_only # unchanged from above: no sensible leakage
        end

        @testset "the filled part of the gap sets D: the water W, within [0, b]" begin
            s = State(grid)
            s.q_x .= state.q_x; s.dhdx .= state.dhdx; s.q_y .= state.q_y; s.dhdy .= state.dhdy
            s.b .= [0.01 * (i + j) for i in 1:nx, j in 1:ny]
            mt_pot_only = MeltTerms{true, true, true, false, true}()
            wd = WithDiffusion(CholeskyDirectSolver(grid))
            copyto!(s.W, s.b); compute_D!(s, p, mt_pot_only, wd) # W = b (FilledCavities): the face mean of b
            D_full = copy(Array(s.D_x)); Dy_full = copy(Array(s.D_y))
            b = Array(s.b)
            @test D_full[2, 1] ≈ (D_eq10(b[2, 1], heat_pot) + D_eq10(b[1, 1], heat_pot)) / 2

            # unfilled cell (2, 1): only W of its gap holds water; an "over-full" cell (3, 1)
            # (W > b, from the water budget) touches at most its gap b
            be = zeros(nx, ny); be[2, 1] = 0.75 * b[2, 1]; be[3, 1] = -0.5 * b[3, 1]
            s.W .= b .- be
            compute_D!(s, p, mt_pot_only, wd)
            @test s.D_x[2, 1] ≈ D_full[2, 1] * (0.25 * b[2, 1] + b[1, 1]) / (b[2, 1] + b[1, 1])
            @test s.D_x[3, 1] ≈ D_full[3, 1] * (0.25 * b[2, 1] + b[3, 1]) / (b[2, 1] + b[3, 1])
            @test s.D_x[4, 1] == D_full[4, 1] # (3, 1) is clamped to its gap b
            @test s.D_y[2, 2] ≈ Dy_full[2, 2] * (0.25 * b[2, 1] + b[2, 2]) / (b[2, 1] + b[2, 2])
            @test s.D_x[2, 2] == D_full[2, 2] # faces not touching (2, 1) are unchanged
        end

        @testset "a channel along x diffuses sideways, and D turns with the grid" begin
            # Flow along x only: the per-face D of old gave the y-faces (q_y = 0) D_y = 0, so a grid-aligned channel had
            # no sideways wall melting. One D per cell makes D_x = D_y on a uniform sheet, and the rotated setup the
            # transposed D.
            wd = WithDiffusion(CholeskyDirectSolver(grid))
            mt_pot_only = MeltTerms{true, true, true, false, true}()
            s = State(grid)
            s.b .= [0.01 + 0.002 * i + 0.001 * j for i in 1:nx, j in 1:ny]
            s.q_x .= -1e-5; s.dhdx .= 0.1
            copyto!(s.W, s.b); compute_D!(s, p, mt_pot_only, wd)
            @test all(>(0), Array(s.D_y)[:, 2:end-1])
            s_uni = State(grid); s_uni.b .= 0.02; s_uni.q_x .= -1e-5; s_uni.dhdx .= 0.1
            copyto!(s_uni.W, s_uni.b); compute_D!(s_uni, p, mt_pot_only, wd)
            @test Array(s_uni.D_y)[2:3, 2:3] ≈ Array(s_uni.D_x)[2:3, 2:3]
            r = State(grid)
            r.b .= permutedims(Array(s.b)); r.q_y .= permutedims(Array(s.q_x)); r.dhdy .= permutedims(Array(s.dhdx))
            copyto!(r.W, r.b); compute_D!(r, p, mt_pot_only, wd)
            @test Array(r.D_y) ≈ permutedims(Array(s.D_x))
            @test Array(r.D_x) ≈ permutedims(Array(s.D_y))
            # only faces between two GROUNDED cells carry D
            s.mask[2, 2] = OTHER_BASIN
            copyto!(s.W, s.b); compute_D!(s, p, mt_pot_only, wd)
            @test s.D_x[2, 2] == 0 && s.D_x[3, 2] == 0 && s.D_y[2, 2] == 0 && s.D_y[2, 3] == 0
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

    @testset "diffusion_source: the -div(D*grad(b)) source term in h's RHS (SUHMO Eq. 11)" begin

        # A tiny 3x3 mask/b/D_x/D_y built by hand (not through Grid/State/set_initial_conditions!)
        # so every neighbour's mask category and every face's D value is exactly controlled --
        # the point of this test is to verify diffusion_source's OWN Neumann-boundary logic
        # (domain edges, OCEAN/LAND neighbours, OTHER_BASIN/FROZEN_BED neighbours), not to
        # exercise the rest of the pipeline (already covered by compute_D!'s own tests above and
        # the end-to-end Simulation test below).
        nx, ny = 3, 3
        dx2, dy2 = 100.0, 100.0 # dx=dy=10, arbitrary but nonzero and distinguishable from 1

        mask = fill(GROUNDED, nx, ny)
        b    = fill(0.05, nx, ny)
        D_x  = zeros(nx + 1, ny)
        D_y  = zeros(nx, ny + 1)

        grid = Grid(4, 4, 1e3, 1e3) # only needed to build a real AbstractLinearSolver for WithDiffusion's `ls` field
        wd = WithDiffusion(CholeskyDirectSolver(grid))

        @testset "NoDiffusion is always exactly zero" begin
            @test Shakti.diffusion_source(Val(false), D_x, D_y, mask, b, 2, 2, nx, ny, dx2, dy2) == 0.0
        end

        @testset "Interior cell: only GROUNDED neighbours contribute, others are zero-flux" begin
            # Cell (2,2)'s four neighbours: east=(3,2) OCEAN, west=(1,2) GROUNDED,
            # north=(2,3) OTHER_BASIN, south=(2,1) GROUNDED. D is deliberately set nonzero at
            # EVERY face, including the OCEAN/OTHER_BASIN ones, so this isolates
            # diffusion_source's own mask check rather than relying on D happening to be zero
            # there (as it would be for OTHER_BASIN/FROZEN_BED in the real pipeline, but NOT for
            # OCEAN/LAND -- see this function's own docstring).
            mask2 = copy(mask)
            mask2[3, 2] = OCEAN
            mask2[2, 3] = OTHER_BASIN

            b2 = copy(b)
            b2[2, 2] = 0.05 # here
            b2[1, 2] = 0.02 # west (GROUNDED, included)
            b2[2, 1] = 0.03 # south (GROUNDED, included)
            b2[3, 2] = 0.10 # east (OCEAN, must be excluded regardless of this value)
            b2[2, 3] = 0.20 # north (OTHER_BASIN, must be excluded regardless of this value)

            D_x2 = copy(D_x)
            D_x2[3, 2] = 1e-8 # east face of cell (2,2) -- nonzero on purpose
            D_x2[2, 2] = 2e-8 # west face of cell (2,2)
            D_y2 = copy(D_y)
            D_y2[2, 3] = 3e-8 # north face of cell (2,2) -- nonzero on purpose
            D_y2[2, 2] = 4e-8 # south face of cell (2,2)

            expected = -(D_x2[2, 2] * (b2[1, 2] - b2[2, 2]) / dx2 + D_y2[2, 2] * (b2[2, 1] - b2[2, 2]) / dy2)
            @test Shakti.diffusion_source(Val(true), D_x2, D_y2, mask2, b2, 2, 2, nx, ny, dx2, dy2) ≈ expected
            @test !(expected ≈ 0.0) # nonzero, driven entirely by the two GROUNDED (included) neighbours

            # Confirm the excluded (OCEAN/OTHER_BASIN) faces really contribute nothing -- wildly
            # changing their D value doesn't move the result at all, proving they're genuinely
            # ignored rather than just coincidentally zero in this particular setup.
            D_x2_alt = copy(D_x2); D_x2_alt[3, 2] = 999.0 # excluded east face
            D_y2_alt = copy(D_y2); D_y2_alt[2, 3] = 999.0 # excluded north face
            @test Shakti.diffusion_source(Val(true), D_x2_alt, D_y2_alt, mask2, b2, 2, 2, nx, ny, dx2, dy2) ≈ expected
        end

        @testset "Domain-edge cell: missing neighbours are zero-flux" begin
            # Corner cell (1,1): no west, no south neighbour at all (out of bounds) -- only
            # east/north can contribute, and only if GROUNDED.
            mask3 = fill(GROUNDED, nx, ny)
            b3 = fill(0.05, nx, ny)
            b3[2, 1] = 0.09 # east neighbour of (1,1)
            b3[1, 2] = 0.07 # north neighbour of (1,1)
            D_x3 = zeros(nx + 1, ny)
            D_x3[2, 1] = 5e-8 # east face of cell (1,1)
            D_y3 = zeros(nx, ny + 1)
            D_y3[1, 2] = 6e-8 # north face of cell (1,1)

            expected_corner = -(D_x3[2, 1] * (b3[2, 1] - b3[1, 1]) / dx2 + D_y3[1, 2] * (b3[1, 2] - b3[1, 1]) / dy2)
            @test Shakti.diffusion_source(Val(true), D_x3, D_y3, mask3, b3, 1, 1, nx, ny, dx2, dy2) ≈ expected_corner
        end

        @testset "M stays symmetric: the source term only touches rhs, never nzval" begin
            # diffusion_source only ever appears added into rhs (see update_SALS_elliptic_kernel!'s
            # own rhs assignment) -- it introduces no new matrix coupling, so h's own operator
            # should be exactly as symmetric/SPD with WithDiffusion as without it.
            sl = RegularizedCoulombSlidingLaw(0.25)
            mask4 = fill(GROUNDED, 4, 4)
            mask4[end, :] .= OCEAN
            mask4[1, :]   .= LAND
            mask4[:, 1]   .= OTHER_BASIN
            A_visc = fill(5e-25, 4, 4)
            zb     = repeat(reshape(-0.02 .* grid.x, 4, 1), 1, 4)
            zs     = zb .+ 500.0
            b0     = [0.01 + 0.002 * i + 0.003 * j for i in 1:4, j in 1:4] # spatially varying, not uniform -- a uniform b would make diffusion_source identically zero everywhere (no gradient to diffuse), which would pass this symmetry check trivially regardless of correctness
            G      = fill(0.06, 4, 4)
            ub_x   = fill(1e-6, 5, 4)
            ub_y   = zeros(4, 5)
            ieb    = zeros(4, 4)
            ieb[3, 3] = 3 / (grid.dx * grid.dy)
            taub_x = zeros(5, 4)
            taub_y = zeros(4, 5)

            s = State(grid)
            p = ModelParameters(e_v = 0.0)
            set_initial_conditions!(s, grid, p, sl, mask4, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            s.D_x .= 1e-8 # nonzero, arbitrary but deterministic, so the term is genuinely exercised
            s.D_y .= 1e-8

            sals = SparseAssembledLinearSystem(grid)
            Shakti.update_SALS_elliptic!(sals, s, grid, p, Arithmetic(), wd)

            @test issymmetric(sals.M)
        end

    end

    @testset "solve_b_diffusion!: b's own coupled diffusion solve (SUHMO Eq. 17)" begin

        # An independent dense-matrix reference solution, built fresh in this test (not by calling
        # into update_SALS_b_diffusion_kernel! itself -- that would just test the code against
        # itself). Small 3x3 grid, mixed mask (GROUNDED interior + OCEAN/LAND/OTHER_BASIN edges) so
        # the Neumann convention at every neighbour category is actually exercised.
        nx, ny = 3, 3
        dx, dy = 10.0, 15.0
        dx2, dy2 = dx^2, dy^2
        dt = 3600.0

        mask = [GROUNDED GROUNDED OCEAN;
                GROUNDED GROUNDED GROUNDED;
                LAND     GROUNDED OTHER_BASIN]

        b      = [0.02 0.03 0.0;  0.015 0.025 0.03; 0.0 0.018 0.04]
        N      = fill(5e5, nx, ny)
        mdot   = fill(1e-8, nx, ny)
        beta   = fill(2e-6, nx, ny)
        abs_ub = fill(1e-6, nx, ny)
        A_visc = fill(5e-25, nx, ny)
        lc     = copy(b) # StandardCreep (l_c = b) for this test
        D_x    = [1e-8 2e-8 0.0; 1.5e-8 2.5e-8 0.5e-8; 0.8e-8 1.2e-8 0.3e-8; 0.0 0.0 0.0] # (nx+1, ny)
        D_y    = [1e-8 2e-8 1.5e-8 0.0; 2e-8 3e-8 2.5e-8 0.5e-8; 0.5e-8 1e-8 0.8e-8 0.0]  # (nx, ny+1)

        p = ModelParameters(e_v = 0.0, n = 3.0, b_min = 0.0) # the hand-built reference below has no b floor (solve_b_diffusion! clamps every cell to [b_min, b_max])
        n_minus_1 = Int(p.n - 1) # match p.n_minus_1_exp's own canonical_exponent path exactly (model_parameters.jl):
                                  # x^2 (Int exponent, power-by-squaring) and x^2.0 (Float exponent, exp(y*log(x)))
                                  # aren't guaranteed bit-identical, so this avoids a spurious ULP-level mismatch
                                  # against the kernel's own fast integer-exponent path

        # Independent reference build: dense (I - dt*div(D*grad(.))) operator and Eq. 17's RHS, with
        # the closure and opening-by-sliding terms implicit (on the diagonal) rather than explicit.
        Nc = nx * ny
        Aref = zeros(Nc, Nc)
        rhsref = zeros(Nc)
        row(i, j) = i + (j - 1) * nx
        for j in 1:ny, i in 1:nx
            r = row(i, j)
            if mask[i, j] == GROUNDED
                aE = (i < nx && mask[i+1, j] == GROUNDED) ? dt * D_x[i+1, j] / dx2 : 0.0
                aW = (i > 1  && mask[i-1, j] == GROUNDED) ? dt * D_x[i, j]   / dx2 : 0.0
                aN = (j < ny && mask[i, j+1] == GROUNDED) ? dt * D_y[i, j+1] / dy2 : 0.0
                aS = (j > 1  && mask[i, j-1] == GROUNDED) ? dt * D_y[i, j]   / dy2 : 0.0
                # Local terms linear in the new b sit on the diagonal (b_diffusion_local_terms):
                # closure C*l_c for C > 0 (l_c/b = 1 here, StandardCreep), and opening by sliding
                # gamma*(br - b) while beta > 0.
                C = A_visc[i, j] * abs(N[i, j])^n_minus_1 * N[i, j]
                gamma = abs_ub[i, j] / p.lr
                @assert C > 0 && beta[i, j] > 0 # both implicit branches are what this reference builds
                Aref[r, r] = 1 + aE + aW + aN + aS + dt * C * (lc[i, j] / b[i, j]) + dt * gamma
                i < nx && (Aref[r, row(i+1, j)] = -aE)
                i > 1  && (Aref[r, row(i-1, j)] = -aW)
                j < ny && (Aref[r, row(i, j+1)] = -aN)
                j > 1  && (Aref[r, row(i, j-1)] = -aS)
                rhsref[r] = b[i, j] + dt * (mdot[i, j] / p.rho_i + gamma * p.br)
            else
                Aref[r, r] = 1.0
                rhsref[r] = b[i, j]
            end
        end
        bref = reshape(Aref \ rhsref, nx, ny)

        @testset "Cholesky solve matches the independent reference exactly" begin
            grid = Grid(nx, ny, dx * (nx - 1), dy * (ny - 1)) # Grid computes dx = Lx/(nx-1), so this reproduces dx/dy above
            @assert grid.dx ≈ dx && grid.dy ≈ dy # sanity: the reference above assumed this exact dx/dy

            s = State(grid)
            s.mask .= mask; s.b .= b; s.N .= N; s.mdot .= mdot; s.beta .= beta; s.abs_ub .= abs_ub
            s.A_visc .= A_visc; s.lc .= lc; s.D_x .= D_x; s.D_y .= D_y; s.bump .= p.br # no uplift: the full bump height (compute_beta!)

            ls = CholeskyDirectSolver(grid)
            solve_b_diffusion!(ls, s, grid, p, dt)

            @test Array(s.b) ≈ bref atol = 1e-12

            # Non-GROUNDED cells are held exactly fixed (frozen identity row) -- b there is
            # unchanged from its input value, not solved toward anything.
            for j in 1:ny, i in 1:nx
                mask[i, j] != GROUNDED && @test s.b[i, j] == b[i, j]
            end
        end

        @testset "CG (SparseAssembledLinearSystem) solve agrees with the Cholesky/reference solution" begin
            grid = Grid(nx, ny, dx * (nx - 1), dy * (ny - 1))
            @assert grid.dx ≈ dx && grid.dy ≈ dy
            s = State(grid)
            s.mask .= mask; s.b .= b; s.N .= N; s.mdot .= mdot; s.beta .= beta; s.abs_ub .= abs_ub
            s.A_visc .= A_visc; s.lc .= lc; s.D_x .= D_x; s.D_y .= D_y; s.bump .= p.br # no uplift: the full bump height (compute_beta!)

            ls = CGIterativeSolver(grid, SparseAssembledLinearSystem)
            solve_b_diffusion!(ls, s, grid, p, dt)

            @test Array(s.b) ≈ bref atol = 1e-6 # CG's own convergence tolerance, looser than Cholesky's exact solve
        end

        @testset "sim.gs is truly ignored under WithDiffusion: different gs, identical result" begin
            # Two Simulations differing ONLY in gap_scheme, both with WithDiffusion active --
            # if gs really has no effect while diffusion is on, one full step_b! call must leave
            # both with bit-identical b.
            grid = Grid(4, 4, 1e3, 1e3)
            sl = RegularizedCoulombSlidingLaw(0.25)
            mask4 = fill(GROUNDED, 4, 4)
            mask4[end, :] .= OCEAN
            mask4[1, :]   .= LAND
            mask4[:, 1]   .= OTHER_BASIN
            A_visc4 = fill(5e-25, 4, 4)
            zb4     = repeat(reshape(-0.02 .* grid.x, 4, 1), 1, 4)
            zs4     = zb4 .+ 500.0
            b04     = [0.01 + 0.002 * i + 0.003 * j for i in 1:4, j in 1:4]
            G4      = fill(0.06, 4, 4)
            ub_x4   = fill(1e-6, 5, 4)
            ub_y4   = zeros(4, 5)
            ieb4    = zeros(4, 4)
            ieb4[3, 3] = 3 / (grid.dx * grid.dy)
            taub_x4 = zeros(5, 4)
            taub_y4 = zeros(4, 5)

            function build(gap_scheme)
                state = State(grid)
                p4 = ModelParameters(e_v = 0.0)
                set_initial_conditions!(state, grid, p4, sl, mask4, A_visc4, zb4, zs4, b04, G4, ub_x4, ub_y4, ieb4, taub_x4, taub_y4)
                state.D_x .= 1e-8; state.D_y .= 1e-8
                ls_h = CholeskyDirectSolver(grid)
                ps = PicardSolver(50, floattype(1e-6), ls_h, grid)
                ls_b = CholeskyDirectSolver(grid)
                return Simulation(grid, state, 1, floattype(3600.0), p4, ConstantMeltInput(), sl; gap_scheme, cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps, diffusion_scheme = WithDiffusion(ls_b))
            end

            sim_explicit = build(ExplicitGapScheme())
            sim_fully_implicit = build(FullyImplicitGapScheme())

            step_b!(sim_explicit)
            step_b!(sim_fully_implicit)

            @test Array(sim_explicit.state.b) == Array(sim_fully_implicit.state.b)
        end

    end

    @testset "ConsistentGapCoupling under WithDiffusion: the head rows store SUHMO's gap change" begin
        # Thin ice with a 1 m^3/s moulin at dt = 1 day, filled cavities: a channel forms within two steps and the wall
        # diffusion matters. The explicit coupling assumes the explicit gap change plus the diffusion of the old gap, and
        # loses or creates up to 1.8 times the melt input of a step; with the gap solve in the Picard loop
        # (DiffusiveGapUpdate) the budget closes to the Picard tolerance, and with unfilled cavities and uplift it creates no water.
        nx, ny = 40, 12
        grid = Grid(nx, ny, 3.9e3, 1.1e3)
        x, y = Array(grid.x), Array(grid.y)
        zb = [0.05 * xi + 15 * sin(2pi * xi / 800) * cos(pi * yj / 1100) for xi in x, yj in y]
        H  = [50 + 200 * sqrt(xi / x[end]) for xi in x, yj in y]
        mask = fill(GROUNDED, nx, ny); mask[1, :] .= LAND
        g = mask .== GROUNDED
        p  = ModelParameters()
        sl = RegularizedCoulombSlidingLaw(0.25)
        ieb = fill(1e-9, nx, ny); ieb[25, 6] += 1.0 / (grid.dx * grid.dy)
        dt = 86400.0
        function moulin_sim(gc; kw...)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, fill(2.4e-24, nx, ny), zb, zb .+ H, fill(1e-3, nx, ny), fill(0.06, nx, ny),
                                    fill(30.0 / (365 * 86400), nx + 1, ny), zeros(nx, ny + 1), ieb, zeros(nx + 1, ny), zeros(nx, ny + 1))
            return Simulation(grid, state, 3, floattype(dt), p, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(),
                              ps = PicardSolver(500, floattype(1e-9), CholeskyDirectSolver(grid), grid),
                              diffusion_scheme = WithDiffusion(CholeskyDirectSolver(grid)), gap_coupling = gc, kw...)
        end
        # worst |(b_new - b_old)/dt + div q - mdot/rho_w - ieb| over three steps, against the grounded source
        function worst_budget(sim)
            worst = 0.0
            for _ in 1:3
                s = sim.state
                b0 = Array(s.b)
                step!(sim)
                @test Shakti.picard_status(sim.hs)[1]
                qx, qy = Array(s.q_x), Array(s.q_y)
                dv = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
                src = Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)
                res = (Array(s.b) .- b0) ./ dt .+ dv .- src
                worst = max(worst, sum(abs, res[g]) / sum(abs, src[g]))
                # GAP-2: the step's water budget (the filled sheet's water is b)
                wc = water_created(sim)
                @test wc.created ≈ sum(res[g]) * dt * grid.dx * grid.dy atol = 1e-9 * wc.input
                @test wc.input ≈ sum(src[g]) * dt * grid.dx * grid.dy
            end
            return worst
        end
        @test worst_budget(moulin_sim(ExplicitGapCoupling())) > 0.1
        @test worst_budget(moulin_sim(ConsistentGapCoupling())) < 5e-5 # set by the Picard tolerance (1e-9 here; 4e-8 at 1e-11)
        @test moulin_sim(nothing).gcp isa ConsistentGapCoupling # the default under WithDiffusion on the Picard solver
        # Without diffusion (filled, NoUplift, b_max = Inf) the moulin drives N < 0, where creep opens the gap. The explicit
        # coupling's head rows assume linear growth, the fully implicit gap scheme grows it as exp(dt*|C|): 1e269 m in step 1,
        # then NaN, which step! now refuses (check_gap_height; at omega = 1e-3 already in step 1). The consistent coupling, the
        # default, keeps it at metres.
        let nod = (; diffusion_scheme = NoDiffusion(), ps = PicardSolver(500, floattype(1e-6), CholeskyDirectSolver(grid), grid))
            ex = moulin_sim(ExplicitGapCoupling(); nod...)
            refused = try
                step!(ex); @test maximum(Array(ex.state.b)[g]) > 1e100
                step!(ex); false
            catch e
                e isa ErrorException
            end
            @test refused
            @test occursin("ConsistentGapCoupling()", Shakti.gap_runaway_hint(ex.gcp))                 # the hint suggests it
            @test !occursin("ConsistentGapCoupling()", Shakti.gap_runaway_hint(ConsistentGapCoupling())) # but not once it is on
            cs = moulin_sim(nothing; nod...)
            @test cs.gcp isa ConsistentGapCoupling # the default without diffusion too
            for _ in 1:3
                step!(cs)
                @test Shakti.picard_status(cs.hs)[1]
            end
            @test maximum(Array(cs.state.b)[g]) < 10
        end
        # LUDirectSolver also solves the gap system, to the same gap as Cholesky
        chol, lu_ = moulin_sim(ConsistentGapCoupling()), moulin_sim(ConsistentGapCoupling(); diffusion_scheme = WithDiffusion(LUDirectSolver(grid)))
        step!(chol); step!(lu_)
        @test Shakti.picard_status(lu_.hs)[1]
        @test maximum(abs, Array(lu_.state.b) .- Array(chol.state.b)) <= 1e-8 * maximum(Array(chol.state.b))
        # Unfilled cavities and uplift: the budget keeps the mismatch in W, but where it would leave a cell less than
        # no water the floor W >= 0 of update_water_budget! creates it (the explicit coupling: 1.2 % of the melt in step 2)
        function worst_created(sim)
            worst = 0.0
            for _ in 1:3
                s = sim.state
                W_old = Array(s.W) # the water at the start of the step
                step!(sim)
                @test Shakti.picard_status(sim.hs)[1]
                qx, qy = Array(s.q_x), Array(s.q_y)
                dv = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
                src = Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)
                W = W_old .+ dt .* (src .- dv)
                worst = max(worst, sum(max.(0, .-W[g])) / dt / sum(abs, src[g]))
                # GAP-2: water_created reports what the floor created
                @test water_created(sim).created ≈ sum(max.(0, .-W[g])) * grid.dx * grid.dy atol = 1e-12 * water_created(sim).input
            end
            return worst
        end
        unf = (; cavity_filling = UnfilledCavities(), uplift = IceUplift(), k_face_choice = "implicit_upwind", ps = PicardSolver(500, floattype(1e-9), LUDirectSolver(grid), grid))
        ex = moulin_sim(ExplicitGapCoupling(); unf...)
        @test worst_created(ex) > 1e-3
        wc = water_created(ex)
        @test wc.total_created > 1e-3 * wc.total_input && wc.total_input > 2 * wc.input # summed over the three steps
        @test worst_created(moulin_sim(ConsistentGapCoupling(); unf...)) < 1e-12
    end

    @testset "ConsistentGapCoupling under WithDiffusion on Helheim (500 m, docs example input)" begin
        # Real data, two 1-day steps from the docs example's initial state: the explicit coupling is already 7 % of the melt
        # off in step 2 (17 % by step 5, growing), the consistent one 5e-5 (Picard tolerance 1e-6).
        m = Module()
        Core.eval(m, :(using Shakti, MAT))
        Base.include(m, joinpath(@__DIR__, "helheim_setup.jl"))
        grid, mask = m.grid, m.mask
        g = mask .== GROUNDED
        p = ModelParameters(rho_i = 917.0, omega = 1e-3, br = 0.0, lr = 2.0, ct = 0.0, mdot_includes_sensible = false)
        dt = 86400.0
        function helheim_budget(gc)
            sim = Simulation(grid, deepcopy(m.state), 2, floattype(dt), p, m.mi, m.sl; cavity_filling = FilledCavities(), uplift = NoUplift(),
                             ps = PicardSolver(500, floattype(1e-6), CholeskyDirectSolver(grid), grid),
                             diffusion_scheme = WithDiffusion(CholeskyDirectSolver(grid)), gap_coupling = gc)
            worst = 0.0
            for _ in 1:2
                s = sim.state
                b0 = Array(s.b)
                step!(sim)
                @test Shakti.picard_status(sim.hs)[1]
                qx, qy = Array(s.q_x), Array(s.q_y)
                dv = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:grid.nx, j in 1:grid.ny]
                src = Array(s.mdot) ./ p.rho_w .+ Array(s.ieb)
                res = (Array(s.b) .- b0) ./ dt .+ dv .- src
                worst = max(worst, sum(abs, res[g]) / sum(abs, src[g]))
                # GAP-2: the step's water budget (the filled sheet's water is b)
                wc = water_created(sim)
                @test wc.created ≈ sum(res[g]) * dt * grid.dx * grid.dy atol = 1e-9 * wc.input
                @test wc.input ≈ sum(src[g]) * dt * grid.dx * grid.dy
            end
            return worst
        end
        @test helheim_budget(ExplicitGapCoupling()) > 0.03
        @test helheim_budget(ConsistentGapCoupling()) < 1e-3
    end
