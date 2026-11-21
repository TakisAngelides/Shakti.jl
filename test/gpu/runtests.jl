# GPU test suite, separate from Pkg.test (which runs on the CPU only): cuDSS against LU, the GPU iterative solvers
# against LU, the cuDSS factorization check, and the CUBLAS reductions of gpu_reductions.jl under GC pressure. Needs an
# NVIDIA GPU. The backend is a compile-time preference, so the suite runs in two environments, one process each:
# env_threads (Threads backend with cuDSS) runs first and writes the LU reference that env_cuda (CUDA backend) reads.
# Usage: julia -t 4 test/gpu/runtests.jl, or sbatch test/gpu/run_gpu_tests.sh on albedo. GPU_STRESS_SECONDS sets the
# length of the reduction stress test (default 60).

const HERE = @__DIR__
const JULIA = Base.julia_cmd()
const NT = Threads.nthreads()

for env in ("env_threads", "env_cuda")
    run(`$JULIA --project=$(joinpath(HERE, env)) -e "using Pkg; Pkg.instantiate()"`)
end

failed = String[]
for (env, file) in (("env_threads", "threads_tests.jl"), ("env_cuda", "cuda_tests.jl"))
    println("\n== $file ($env, $NT threads)")
    p = run(ignorestatus(`$JULIA -t $NT --project=$(joinpath(HERE, env)) $(joinpath(HERE, file))`))
    success(p) || push!(failed, file)
end
isempty(failed) || error("GPU tests failed in ", join(failed, ", "))
println("\nGPU tests passed")
