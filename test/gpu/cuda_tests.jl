# CUDA backend: the GPU iterative solvers against the LU reference of threads_tests.jl, and the CUBLAS reductions of
# gpu_reductions.jl, including a stress test under the GC pressure that made the unpreserved dot fail.
include("common.jl")

Shakti.backend == "CUDA" || error("cuda_tests.jl needs the CUDA backend (env_cuda)")
const CUDA = Shakti.CUDA
const Krylov = Shakti.Krylov
println("CUDA backend, ", Threads.nthreads(), " threads, ", CUDA.name(CUDA.device()))

loadref(sheet) = (f = joinpath(RESULTS, "lu_$sheet.jls");
                  isfile(f) ? deserialize(f) : error("$f missing: run threads_tests.jl first (runtests.jl does)"))

@testset "GPU, CUDA backend" begin
    @testset "Reductions dispatch and agree with the CPU" begin
        for T in (Float32, Float64)
            x, y = rand(T, 10_001), rand(T, 10_001)
            xg, yg = CUDA.CuVector(x), CUDA.CuVector(y)
            @test which(Krylov.kdot, typeof.((1, xg, yg))).module === Shakti
            @test which(Krylov.knorm, typeof.((1, xg))).module === Shakti
            @test Krylov.kdot(length(x), xg, yg) ≈ dot(x, y) rtol = 10eps(T)
            @test Krylov.kdotr(length(x), xg, yg) ≈ dot(x, y) rtol = 10eps(T)
            @test Krylov.knorm(length(x), xg) ≈ norm(x) rtol = 10eps(T)
            @test Shakti.safe_dot(xg, yg) ≈ dot(x, y) rtol = 10eps(T)
        end
    end

    # The failing pattern of the CG runs (PostDoc_Latex_Notes/Notes_on_Shakti/cuda_dot_curef_gc_race.tex): host garbage
    # on every other thread and small device allocations between bursts of dots. CUDA.CUBLAS.dot fails within a minute
    # of this on an A100 (Julia 1.13, CUDA.jl 5.11.3, -t 4); the preserved reductions must not.
    @testset "Reductions under GC pressure" begin
        secs = parse(Float64, get(ENV, "GPU_STRESS_SECONDS", "60"))
        Threads.nthreads() > 1 || @warn "the stress test needs several threads (julia -t 4) to reproduce the race"
        n = 41_000
        x, y = CUDA.rand(Float64, n), CUDA.rand(Float64, n)
        expected = (dot(Array(x), Array(y)), norm(Array(x)))
        stop = Threads.Atomic{Bool}(false)
        workers = [Threads.@spawn begin
                       while !stop[]
                           v = [rand(2000) for _ in 1:200]; v = nothing
                           yield()
                       end
                   end for _ in 2:Threads.nthreads()]
        npairs, wrong, err = 0, 0, nothing
        t0 = time()
        try
            while time() - t0 < secs
                for _ in 1:2000
                    d, r = Krylov.kdot(n, x, y), Krylov.knorm(n, x)
                    npairs += 1
                    (isapprox(d, expected[1]; rtol = 1e-10) && isapprox(r, expected[2]; rtol = 1e-10)) || (wrong += 1)
                end
                for _ in 1:20; CUDA.zeros(Float64, 1000); end
            end
        catch e
            err = e
        finally
            stop[] = true
            foreach(wait, workers)
        end
        println("  $npairs dot and norm pairs in ", round(time() - t0; digits = 1), " s", err === nothing ? "" : ", failed: " * sprint(showerror, err))
        @test err === nothing
        @test wrong == 0
    end

    @testset "BiCGStab with multigrid against LU, $sheet sheet" for sheet in (:default, :spd)
        test_against(test_sheet(g -> BiCGStabIterativeSolver(g; multigrid = true), sheet), loadref(sheet), 1e-3; label = "BiCGStab")
    end

    @testset "CG against LU, spd sheet" begin
        ref = loadref(:spd)
        test_against(test_sheet(g -> CGIterativeSolver(g, MatrixFreeLinearSystem; multigrid = true), :spd), ref, 1e-3; label = "CG multigrid")
        # Chebyshev with Lanczos bounds runs Shakti's own GPU dots (estimate_eigenvalue_bounds!, through safe_dot)
        test_against(test_sheet(g -> CGIterativeSolver(g, MatrixFreeLinearSystem; multigrid = false, chebyshev_degree = 4,
                                                       chebyshev_bounds = :lanczos), :spd), ref, 1e-3; label = "CG Chebyshev")
    end
end
