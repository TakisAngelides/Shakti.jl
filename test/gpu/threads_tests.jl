# Threads backend with cuDSS: CUDSSDirectSolver against LUDirectSolver on the test sheet, the factorization check of
# cudss_solve!, and the LU reference for cuda_tests.jl (results/lu_<sheet>.jls).
include("common.jl")
using SparseArrays

Shakti.backend == "Threads" && Shakti.use_cudss || error("threads_tests.jl needs the Threads backend with cudss = true (env_threads)")
println("Threads backend, cuDSS, ", Threads.nthreads(), " threads, ", Shakti.CUDA.name(Shakti.CUDA.device()))

@testset "GPU, Threads backend" begin
    @testset "cuDSS against LU, $sheet sheet" for sheet in (:default, :spd)
        ref = test_sheet(LUDirectSolver, sheet)
        @test ref.unconverged == 0
        serialize(joinpath(RESULTS, "lu_$sheet.jls"), ref)
        test_against(test_sheet(g -> CUDSSDirectSolver(g; structure = :general), sheet), ref, 1e-6; label = "cuDSS LU")
        sheet == :spd && test_against(test_sheet(g -> CUDSSDirectSolver(g; structure = :spd), sheet), ref, 1e-6; label = "cuDSS Cholesky")
    end

    # A diagonally dominant matrix on the solver's own stencil pattern, then the same with one cell's row and column
    # zeroed (the all-dry cell of b = 0, see ModelParameters), then the first matrix again.
    @testset "cuDSS factorization check, $structure" for structure in (:general, :spd)
        g = Grid(12, 8, 11e3, 7e3)
        ls = CUDSSDirectSolver(g; structure)
        M = ls.sals.M
        n = size(M, 1)
        good = similar(M.nzval)
        for j in 1:n, k in nzrange(M, j)
            good[k] = M.rowval[k] == j ? 5.0 : -1.0
        end
        rhs = collect(range(1.0, 2.0; length = n))
        solve(vals) = (M.nzval .= vals; ls.sals.rhs .= rhs; copy(Shakti.cudss_solve!(ls)))
        ref = SparseMatrixCSC(M.m, M.n, M.colptr, M.rowval, good) \ rhs
        @test solve(good) ≈ ref rtol = 1e-12

        c = n ÷ 2 + 3
        bad = copy(good)
        for j in 1:n, k in nzrange(M, j)
            (j == c || M.rowval[k] == c) && (bad[k] = 0.0)
        end
        @test_throws (structure == :general ? SingularException : PosDefException) solve(bad)
        @test solve(good) ≈ ref rtol = 1e-12             # the check leaves the solver usable on the next matrix
        @test_throws (structure == :general ? SingularException : PosDefException) solve(bad)
        @test solve(good) ≈ ref rtol = 1e-12
    end
end
