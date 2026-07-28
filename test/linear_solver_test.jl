    @testset "MatrixFreeLinearSystem agrees with SparseAssembledLinearSystem" begin

        # A nontrivial mask/state: GROUNDED interior, OCEAN/LAND/OTHER_BASIN
        # edges, sloped bed, point-source moulin -- exercises every branch of
        # update_SALS_kernel!/update_MFLS_kernel! (all four mask cases, and
        # every face of the GROUNDED stencil), not just a trivial uniform case
        # that could pass by coincidence.
        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        state = State(grid)
        p = ModelParameters(e_v = 0.0)
        sl = RegularizedCoulombSlidingLaw(0.25)
        kfs = Arithmetic()

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
        taub_x = zeros(nx + 1, ny) # unused by RegularizedCoulombSlidingLaw (recomputed from N/ub each Picard iteration)
        taub_y = zeros(nx, ny + 1)

        set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

        # Perturb h and recompute everything it feeds, the same sequence
        # Picard_iteration! runs (minus the linear solve itself) -- so the
        # assembly is compared on a state that actually looks like a mid-solve
        # iterate, not just the untouched initial condition.
        state.h .+= 0.01 .* reshape(1:(nx * ny), nx, ny)
        compute_dhdx!(state, grid)
        compute_dhdy!(state, grid)
        compute_pw!(state, p)
        compute_dpwdx!(state, grid)
        compute_dpwdy!(state, grid)
        compute_N!(state)
        compute_q_x!(state, p)
        compute_q_y!(state, p)
        compute_Re_x!(state, p)
        compute_Re_y!(state, p)
        compute_Re!(state)
        compute_taub_x!(state, p, sl)
        compute_taub_y!(state, p, sl)
        shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
        compute_mdot!(state, p, shs)
        compute_K!(state, p)

        sals = SparseAssembledLinearSystem(grid)
        mfls = MatrixFreeLinearSystem(grid)

        Shakti.update_SALS!(sals, state, grid, p, kfs)
        Shakti.update_MFLS!(mfls, state, grid, p, kfs)

        @test sals.rhs ≈ mfls.rhs

        # Dirichlet BCs (OCEAN/LAND) are eliminated symmetrically -- a
        # GROUNDED cell's coupling to a Dirichlet neighbour is folded into
        # rhs rather than left as a one-sided matrix entry -- so the
        # assembled operator should be exactly symmetric.
        @test issymmetric(sals.M)

        # SALS bakes the minus sign for off-diagonal (neighbor) entries
        # directly into nzval; MFLS stores the raw positive face conductance
        # and the minus sign is applied later, in the matvec -- so the
        # diagonal compares directly, but each off-diagonal comparison needs
        # a sign flip.
        for j in 1:ny, i in 1:nx
            @test sals.M.nzval[sals.idxP[i, j]] ≈ mfls.aP[i, j]
            i < nx && @test sals.M.nzval[sals.idxE[i, j]] ≈ -mfls.aE[i, j]
            i > 1  && @test sals.M.nzval[sals.idxW[i, j]] ≈ -mfls.aW[i, j]
            j < ny && @test sals.M.nzval[sals.idxN[i, j]] ≈ -mfls.aN[i, j]
            j > 1  && @test sals.M.nzval[sals.idxS[i, j]] ≈ -mfls.aS[i, j]
        end

    end

    @testset "CGIterativeSolver matches CholeskyDirectSolver" begin

        # Same nontrivial mask/state as above, but this time actually solving
        # (not just comparing assembly) -- CG and Cholesky are only valid
        # because the assembled operator is SPD (see the symmetry test
        # above), so this confirms the swap from GMRES/BiCGSTAB/LU to
        # CG/Cholesky still converges to/produces the same head field.
        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        p = ModelParameters(e_v = 0.0)
        sl = RegularizedCoulombSlidingLaw(0.25)
        kfs = Arithmetic()

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
        taub_x = zeros(nx + 1, ny) # unused by RegularizedCoulombSlidingLaw (recomputed from N/ub each Picard iteration)
        taub_y = zeros(nx, ny + 1)

        function fresh_state()
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            state.h .+= 0.01 .* reshape(1:(nx * ny), nx, ny)
            compute_dhdx!(state, grid)
            compute_dhdy!(state, grid)
            compute_pw!(state, p)
            compute_dpwdx!(state, grid)
            compute_dpwdy!(state, grid)
            compute_N!(state)
            compute_q_x!(state, p)
            compute_q_y!(state, p)
            compute_Re_x!(state, p)
            compute_Re_y!(state, p)
            compute_Re!(state)
            compute_taub_x!(state, p, sl)
            compute_taub_y!(state, p, sl)
            shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
            compute_mdot!(state, p, shs)
            compute_K!(state, p)
            return state
        end

        state_lu = fresh_state()
        ls_lu = CholeskyDirectSolver(grid)
        Shakti.solve_elliptic_linear_system!(ls_lu, state_lu, grid, p, kfs)

        # amg = true is CGIterativeSolver(g, SparseAssembledLinearSystem)'s
        # default (see linear_solver.jl) -- locking that in here so a future
        # default change doesn't silently drift unnoticed.
        state_cg_default = fresh_state()
        ls_cg_default = CGIterativeSolver(grid, SparseAssembledLinearSystem)
        @test ls_cg_default.precond isa AMGPreconditioner
        Shakti.solve_elliptic_linear_system!(ls_cg_default, state_cg_default, grid, p, kfs)
        @test state_cg_default.h ≈ state_lu.h atol=1e-5

        # Plain Jacobi (amg = false, chebyshev_degree = nothing): the baseline
        # every accelerated preconditioner below should beat-or-match on CG
        # iteration count.
        state_cg_jacobi = fresh_state()
        ls_cg_jacobi = CGIterativeSolver(grid, SparseAssembledLinearSystem; amg = false)
        Shakti.solve_elliptic_linear_system!(ls_cg_jacobi, state_cg_jacobi, grid, p, kfs)
        @test state_cg_jacobi.h ≈ state_lu.h atol=1e-5

        state_cg_mf = fresh_state()
        ls_cg_mf = CGIterativeSolver(grid, MatrixFreeLinearSystem) # MFLS has no amg option (see linear_solver.jl); this is plain Jacobi
        Shakti.solve_elliptic_linear_system!(ls_cg_mf, state_cg_mf, grid, p, kfs)
        @test state_cg_mf.h ≈ state_lu.h atol=1e-5

        # ChebyshevPreconditioner (see preconditioner.jl): should converge to
        # the same head field as plain Jacobi/Cholesky regardless of degree
        # (it's still an exact solve to CG's tolerance, just via a different
        # preconditioner), and should need fewer outer CG iterations than
        # Jacobi once the degree is large enough to approximate A^-1 well.
        state_cg_cheb = fresh_state()
        ls_cg_cheb = CGIterativeSolver(grid, SparseAssembledLinearSystem; chebyshev_degree = 4, amg = false)
        Shakti.solve_elliptic_linear_system!(ls_cg_cheb, state_cg_cheb, grid, p, kfs)
        @test state_cg_cheb.h ≈ state_lu.h atol=1e-5
        @test ls_cg_cheb.ws.stats.niter <= ls_cg_jacobi.ws.stats.niter

        state_cg_mf_cheb = fresh_state()
        ls_cg_mf_cheb = CGIterativeSolver(grid, MatrixFreeLinearSystem; chebyshev_degree = 4)
        Shakti.solve_elliptic_linear_system!(ls_cg_mf_cheb, state_cg_mf_cheb, grid, p, kfs)
        @test state_cg_mf_cheb.h ≈ state_lu.h atol=1e-5
        @test ls_cg_mf_cheb.ws.stats.niter <= ls_cg_mf.ws.stats.niter

        # AMGPreconditioner (see preconditioner.jl): SALS-only (no GPU array
        # support in AlgebraicMultigrid.jl). Measured (test/amg_rerun.jl,
        # test/precond_vs_channelization.jl) to consistently beat plain
        # Jacobi's CG iteration count by a wide margin, hence the default.
        state_cg_amg = fresh_state()
        ls_cg_amg = CGIterativeSolver(grid, SparseAssembledLinearSystem; amg = true)
        Shakti.solve_elliptic_linear_system!(ls_cg_amg, state_cg_amg, grid, p, kfs)
        @test state_cg_amg.h ≈ state_lu.h atol=1e-5
        @test ls_cg_amg.ws.stats.niter <= ls_cg_jacobi.ws.stats.niter

        # amg and chebyshev_degree are mutually exclusive preconditioner
        # choices, and amg isn't offered on MatrixFreeLinearSystem at all.
        @test_throws ErrorException CGIterativeSolver(grid, SparseAssembledLinearSystem; amg = true, chebyshev_degree = 4)
        @test_throws ErrorException CGIterativeSolver(grid, MatrixFreeLinearSystem; amg = true)

    end

    @testset "ChebyshevPreconditioner degrades gracefully on invalid eigenvalue estimate" begin

        # The Lanczos-via-CG-recurrence eigenvalue estimator (estimate_eigenvalue_bounds)
        # is only exact in infinite precision; on ill-conditioned problems it
        # can hand back a nonsensical (even negative) bounds interval instead
        # of erroring (see test/cheb_512_diag.jl, which found exactly this at
        # 512x512 -- too large a grid for a fast unit test, but an all-zero
        # rhs reproduces the same NaN/Inf cascade cheaply and deterministically:
        # gamma = dot(r,r) = 0 makes the very first alphas[1] = gamma/pAp = 0/0).
        # update_chebyshev_bounds! should catch this and keep the previous
        # (here, still-placeholder) bounds rather than propagating NaN/Inf
        # into P.lambda_min/lambda_max, which downstream would either silently
        # corrupt the preconditioner or surface as Krylov.jl's much more
        # cryptic "operator or preconditioner is not SPD" error.
        n = 36
        M = spdiagm(0 => fill(4.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-1.0, n - 1))
        d = fill(4.0, n)
        cheb = ChebyshevPreconditioner(M, d, 4)
        @test cheb.lambda_min == 1.0 && cheb.lambda_max == 1.0 # construction-time placeholder

        bad_rhs = zeros(n)
        @test_logs (:warn,) match_mode = :any update_chebyshev_bounds!(cheb, bad_rhs)
        @test cheb.lambda_min == 1.0 && cheb.lambda_max == 1.0 # unchanged, not NaN/Inf

    end
