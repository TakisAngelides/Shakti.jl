# GPU-accelerated direct sparse Cholesky solver via CUDSS.jl (NVIDIA's cuDSS). Only ever included
# under "Threads" or "CUDA" backend (see Shakti.jl's own `@static if` guard around this file's
# `include`) -- CUDA/CUDSS symbols are freely used at this file's top level below, unlike scripts
# that need to run under any backend and therefore gate CUDA-specific code with their own
# `@static if` internally (see e.g. test/benchmarks/benchmark_memory.jl).
#
# REQUIRES "Threads" backend at construction (like CholeskyDirectSolver, NOT "CUDA"): assembly
# (SparseAssembledLinearSystem/update_SALS_elliptic!) is fundamentally CPU-only -- it builds a
# plain SparseMatrixCSC via a Dict-based index map, which cannot be written to from within a
# GPU-native kernel -- so State/Grid must stay CPU-resident (ParallelStencil's "Threads" backend),
# exactly as CholeskyDirectSolver already requires. Under "CUDA" backend, ParallelStencil would
# make State's own fields GPU-resident CuArrays instead, which update_SALS_elliptic! cannot
# consume at all (confirmed directly: this was tried first and failed with a CUDA kernel-compile
# error deep inside the assembly kernel). CUDSSDirectSolver uses CUDA/CUDSS purely as an
# accelerator library for its OWN private GPU arrays -- decoupled from ParallelStencil's backend
# entirely, exactly like calling any other GPU-aware library from ordinary CPU-resident code.
#
# CRITICAL correctness note, learned the hard way (see beyond_solver_choice.tex's CUDSS section for
# the full story): CUDSS's `cholesky!` refactorize path TRUSTS that its input matrix has EXACTLY
# the same sparsity pattern as whatever was originally analyzed, and does not validate this --
# silently producing wrong answers (not an error) if the pattern has drifted even slightly. This
# happened in initial testing via an innocent-looking `M .* 1.0001` perturbation, which silently
# dropped explicit stored zeros (Julia's sparse arithmetic broadcast optimizes these away). The
# fix, and the discipline this solver follows throughout: NEVER reconstruct the GPU matrix from a
# fresh CPU-side conversion after the first build -- every solve updates the EXISTING GPU matrix's
# `nzVal` array in place, via a one-time-computed index permutation (`vals_perm`) from `sals.M`'s
# CSC nonzero order to the GPU CSR's nonzero order. `vals_perm` is built via a "marker" trick (fill
# `nzval` with unique sequential integers, convert to CSR once, read back which original index
# landed in each CSR slot) specifically so this doesn't need to understand or rely on CUDA.jl's
# internal CSC->CSR reordering.
#
# sals.M is passed to CUDSS as the FULL matrix (`view = 'F'`), not `triu(sals.M)`: it already
# stores both triangles explicitly (see SparseAssembledLinearSystem's own docstring -- Dirichlet
# neighbours are eliminated symmetrically, so both (i,j) and (j,i) are genuinely present), so no
# triangular-extraction step -- and therefore none of the pattern-preservation risk a fresh
# extraction would reintroduce -- is needed at all.
#
# Architecturally mirrors CholeskyDirectSolver otherwise: same SparseAssembledLinearSystem-based
# assembly, same "factorize once, refactorize every solve reusing the symbolic analysis" discipline,
# just kept on the GPU throughout. Scoped to the elliptic head equation only (like
# NewtonJFNKSolver), not the parabolic/b-diffusion solves CholeskyDirectSolver also implements --
# add those the same way if/when a GPU-backed parabolic or diffusion solve is actually needed.

"""
$(TYPEDSIGNATURES)

GPU direct sparse Cholesky solver via CUDSS.jl, CUDA backend only -- errors at construction time
under any other backend. See this file's module-level notes for the correctness discipline this
implementation follows (never reconstruct the GPU matrix fresh; update its `nzVal` in place via a
precomputed permutation).
"""
mutable struct CUDSSDirectSolver{F <: AbstractFloat, SALS <: SparseAssembledLinearSystem, AGPU, SLV, VGPU <: AbstractVector{F}} <: AbstractDirectSolver
    sals::SALS
    A_gpu::AGPU
    vals_perm::Vector{Int} # vals_perm[k] = index into sals.M.nzval whose value belongs at A_gpu.nzVal[k]
    solver::SLV            # CUDSS factorization object (from `cholesky(A_gpu; view='F')`)
    b_gpu::VGPU
    x_gpu::VGPU
    gpu_vals::VGPU          # GPU-side scratch for the permuted nzval, length nnz(sals.M), reused every solve
    h_vec::Vector{F}        # CPU-side result buffer, mirrors CholeskyDirectSolver's h_vec
end

"""
$(TYPEDSIGNATURES)

Builds a [`CUDSSDirectSolver`](@ref) on grid `g`. Requires the "Threads" backend at construction
(like [`CholeskyDirectSolver`](@ref)) -- see this file's module-level notes for why "CUDA" backend
does not work here (ParallelStencil's own GPU field arrays are incompatible with
`SparseAssembledLinearSystem`'s CPU-only assembly).
"""
function CUDSSDirectSolver(g::Grid{F}) where F

    backend != "Threads" && error("CUDSSDirectSolver requires the \"Threads\" backend (current backend: $backend) -- it uses CUDA/CUDSS internally regardless, see this file's module-level notes.")

    sals = SparseAssembledLinearSystem(g)
    n = g.nx * g.ny
    nz = nnz(sals.M)

    # One-time "marker" trick to learn the CSC(CPU)->CSR(GPU) value-index permutation without
    # depending on CUDA.jl's internal conversion implementation: fill nzval with unique sequential
    # markers, convert to CSR once, read back which original index landed in each CSR slot.
    marker = copy(sals.M)
    marker.nzval .= 1:nz
    A_gpu_marker = CuSparseMatrixCSR{F, Int32}(marker)
    vals_perm = Int.(Array(A_gpu_marker.nzVal))

    A_gpu = CuSparseMatrixCSR{F, Int32}(sals.M) # real (placeholder, at this point) initial values
    solver = cholesky(A_gpu; view = 'F')

    b_gpu    = CUDA.zeros(F, n)
    x_gpu    = CUDA.zeros(F, n)
    gpu_vals = CUDA.zeros(F, nz)
    h_vec    = zeros(F, n)

    return CUDSSDirectSolver(sals, A_gpu, vals_perm, solver, b_gpu, x_gpu, gpu_vals, h_vec)
end

"""
$(TYPEDSIGNATURES)

Solves the elliptic head equation's linear system under [`CUDSSDirectSolver`](@ref): rebuilds
`sals.M`/`sals.rhs` on the CPU (same [`update_SALS_elliptic!`](@ref) call as
[`CholeskyDirectSolver`](@ref)), scatters the updated values into the existing, fixed-pattern GPU
matrix in place, refactorizes on the GPU reusing the symbolic analysis, solves, and copies the
result back into `s.h`.
"""
function solve_elliptic_linear_system!(ls::CUDSSDirectSolver, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, ds::AbstractDiffusionScheme = NoDiffusion())

    update_SALS_elliptic!(ls.sals, s, g, p, kfs, ds) # refresh sals.M/rhs on CPU

    host_vals = ls.sals.M.nzval[ls.vals_perm] # CPU-side gather into the GPU matrix's own nonzero order
    copyto!(ls.gpu_vals, host_vals)
    copyto!(ls.A_gpu.nzVal, ls.gpu_vals)

    cholesky!(ls.solver, ls.A_gpu) # refactorize in place, reusing the symbolic analysis

    copyto!(ls.b_gpu, ls.sals.rhs)
    ls.x_gpu .= ls.solver \ ls.b_gpu

    copyto!(ls.h_vec, ls.x_gpu)
    s.h .= reshape(ls.h_vec, g.nx, g.ny)

    return s
end
