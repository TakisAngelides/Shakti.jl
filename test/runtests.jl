using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using Test
using LinearAlgebra

@testset "Shakti.jl" begin

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
        mi = ConstantMeltInput()
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

        set_initial_conditions!(state, grid, p, mi, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)

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
        compute_taub_x!(state, p)
        compute_taub_y!(state, p)
        shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
        compute_mdot!(state, p, shs)
        compute_K!(state, p)

        sals = SparseAssembledLinearSystem(grid)
        mfls = MatrixFreeLinearSystem(grid)

        Shakti.update_SALS!(sals, state, grid, p, kfs, mi)
        Shakti.update_MFLS!(mfls, state, grid, p, kfs, mi)

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
        mi = ConstantMeltInput()
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

        function fresh_state()
            state = State(grid)
            set_initial_conditions!(state, grid, p, mi, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)
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
            compute_taub_x!(state, p)
            compute_taub_y!(state, p)
            shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
            compute_mdot!(state, p, shs)
            compute_K!(state, p)
            return state
        end

        state_lu = fresh_state()
        ls_lu = CholeskyDirectSolver(grid)
        Shakti.solve_linear_system!(ls_lu, state_lu, grid, p, kfs, mi)

        state_cg = fresh_state()
        ls_cg = CGIterativeSolver(grid, SparseAssembledLinearSystem)
        Shakti.solve_linear_system!(ls_cg, state_cg, grid, p, kfs, mi)
        @test state_cg.h ≈ state_lu.h atol=1e-5

        state_cg_mf = fresh_state()
        ls_cg_mf = CGIterativeSolver(grid, MatrixFreeLinearSystem)
        Shakti.solve_linear_system!(ls_cg_mf, state_cg_mf, grid, p, kfs, mi)
        @test state_cg_mf.h ≈ state_lu.h atol=1e-5

        # ChebyshevPreconditioner (see preconditioner.jl): should converge to
        # the same head field as plain Jacobi/Cholesky regardless of degree
        # (it's still an exact solve to CG's tolerance, just via a different
        # preconditioner), and should need fewer outer CG iterations than
        # Jacobi once the degree is large enough to approximate A^-1 well.
        state_cg_cheb = fresh_state()
        ls_cg_cheb = CGIterativeSolver(grid, SparseAssembledLinearSystem; chebyshev_degree = 4)
        Shakti.solve_linear_system!(ls_cg_cheb, state_cg_cheb, grid, p, kfs, mi)
        @test state_cg_cheb.h ≈ state_lu.h atol=1e-5
        @test ls_cg_cheb.ws.stats.niter <= ls_cg.ws.stats.niter

        state_cg_mf_cheb = fresh_state()
        ls_cg_mf_cheb = CGIterativeSolver(grid, MatrixFreeLinearSystem; chebyshev_degree = 4)
        Shakti.solve_linear_system!(ls_cg_mf_cheb, state_cg_mf_cheb, grid, p, kfs, mi)
        @test state_cg_mf_cheb.h ≈ state_lu.h atol=1e-5
        @test ls_cg_mf_cheb.ws.stats.niter <= ls_cg_mf.ws.stats.niter

    end

end
