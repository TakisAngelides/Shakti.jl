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
# just kept on the GPU throughout. Solves the head system and SUHMO's gap
# system (solve_b_diffusion!): both assemble values into the same fixed pattern of sals.M,
# so one vals_perm and one symbolic analysis serve them (cudss_solve!).

"""
$(TYPEDSIGNATURES)

GPU direct sparse solver via CUDSS.jl (Cholesky with `structure = :spd`, LU with `:general` for the non-symmetric
`"implicit_upwind"` matrix). Requires the `"Threads"` backend: the fields stay on the CPU and only the matrix and the
solve go to the GPU; errors at construction time under any other backend. See this file's module-level notes for the correctness discipline this
implementation follows (never reconstruct the GPU matrix fresh; update its `nzVal` in place via a
precomputed permutation).
"""
mutable struct CUDSSDirectSolver{F <: AbstractFloat, SALS <: SparseAssembledLinearSystem, AGPU, SLV, VGPU <: AbstractVector{F}} <: AbstractDirectSolver
    sals::SALS
    A_gpu::AGPU
    vals_perm::Vector{Int} # vals_perm[k] = index into sals.M.nzval whose value belongs at A_gpu.nzVal[k]
    solver::SLV            # CUDSS factorization object (from `cholesky(A_gpu; view='F')`, or `lu(A_gpu)` under structure = :general)
    general::Bool          # structure = :general: LU of a non-symmetric matrix (ImplicitUpwind) instead of Cholesky
    b_gpu::VGPU
    x_gpu::VGPU
    host_vals::Vector{F}    # CPU-side scratch for the permuted nzval, length nnz(sals.M), reused every solve (no allocation per solve)
    h_vec::Vector{F}        # CPU-side result buffer, mirrors CholeskyDirectSolver's h_vec
end

"""
$(TYPEDSIGNATURES)

Builds a [`CUDSSDirectSolver`](@ref) on grid `g`. Requires the "Threads" backend at construction
(like [`CholeskyDirectSolver`](@ref)) -- see this file's module-level notes for why "CUDA" backend
does not work here (ParallelStencil's own GPU field arrays are incompatible with
`SparseAssembledLinearSystem`'s CPU-only assembly). `structure`: `:general` (default, LU: the GPU counterpart of
[`LUDirectSolver`](@ref), for the non-symmetric matrix [`ImplicitUpwind`](@ref) and so the default sheet assembles) or
`:spd` (Cholesky, symmetric matrices only: SHAKTI's sheet, `"upwind"`, or WithDiffusion's gap solve).
"""
function CUDSSDirectSolver(g::Grid{F}; structure::Symbol = :general) where F
    structure in (:spd, :general) || error("Unknown structure: $structure (use :spd or :general)")

    backend != "Threads" && error("CUDSSDirectSolver requires the \"Threads\" backend (current backend: $backend) -- it uses CUDA/CUDSS internally regardless, see this file's module-level notes.")

    sals = SparseAssembledLinearSystem(g)
    n = g.nx * g.ny
    nz = nnz(sals.M)

    # One-time "marker" trick to learn the CSC(CPU)->CSR(GPU) value-index permutation without
    # depending on CUDA.jl's internal conversion implementation: fill nzval with unique sequential
    # markers, convert to CSR once, read back which original index landed in each CSR slot. The
    # markers are Float64 whatever F is: Float32 holds integers exactly only up to 2^24 (~16.8M
    # nonzeros, ~3.3M cells of the 5-point stencil), beyond which markers would collide and values
    # would be scattered into wrong slots. Float64 is exact up to 2^53. Built from sals.M's own
    # colptr/rowval, so every stored entry (explicit zeros included) gets a marker.
    marker = SparseMatrixCSC(size(sals.M)..., copy(sals.M.colptr), copy(sals.M.rowval), Float64.(1:nz))
    A_gpu_marker = CuSparseMatrixCSR{Float64, Int32}(marker)
    vals_perm = round.(Int, Array(A_gpu_marker.nzVal))
    A_gpu_marker = nothing
    length(vals_perm) == nz && isperm(vals_perm) ||
        error("CUDSSDirectSolver: the CSC -> CSR value permutation is not a permutation of the $nz stored entries")

    A_gpu = CuSparseMatrixCSR{F, Int32}(sals.M) # real (placeholder, at this point) initial values
    general = structure == :general
    solver = general ? lu(A_gpu) : cholesky(A_gpu; view = 'F')

    b_gpu    = CUDA.zeros(F, n)
    x_gpu    = CUDA.zeros(F, n)
    host_vals = zeros(F, nz)
    h_vec     = zeros(F, n)

    return CUDSSDirectSolver(sals, A_gpu, vals_perm, solver, general, b_gpu, x_gpu, host_vals, h_vec)
end

supports_nonsymmetric(ls::CUDSSDirectSolver) = ls.general

"""
$(TYPEDSIGNATURES)

Solves the elliptic head equation's linear system under [`CUDSSDirectSolver`](@ref): rebuilds
`sals.M`/`sals.rhs` on the CPU (same `update_SALS_elliptic!` call as
[`CholeskyDirectSolver`](@ref)), scatters the updated values into the existing, fixed-pattern GPU
matrix in place, refactorizes on the GPU reusing the symbolic analysis, solves, and copies the
result back into `s.h`.
"""
function solve_elliptic_linear_system!(ls::CUDSSDirectSolver, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, ds::AbstractDiffusionScheme = NoDiffusion(), gc::AbstractGapCoupling = ExplicitGapCoupling(), dt = 0, be_old = s.b_e)
    update_SALS_elliptic!(ls.sals, s, g, p, kfs, ds, gc, dt, be_old) # refresh sals.M/rhs on CPU
    cudss_solve!(ls)
    s.h .= reshape(ls.h_vec, g.nx, g.ny)
    return s
end

"""
$(TYPEDSIGNATURES)

Solves SUHMO's gap system under [`CUDSSDirectSolver`](@ref) (the second solver of [`WithDiffusion`](@ref)), clamped
into `out` as the [`CholeskyDirectSolver`](@ref) method. `update_SALS_b_diffusion!` writes values only into
the fixed pattern of `ls.sals.M` (the same stencil slots as the head system), so the precomputed `vals_perm` and the
symbolic analysis stay valid.
"""
function solve_b_diffusion!(ls::CUDSSDirectSolver, s::State, g::Grid, p::ModelParameters, dt, out = s.b)
    update_SALS_b_diffusion!(ls.sals, s, g, p, dt)
    cudss_solve!(ls)
    out .= clamp.(reshape(ls.h_vec, g.nx, g.ny), p.b_min, p.b_max)
end

# Scatters the freshly assembled sals.M values into the existing GPU matrix in place (never a fresh conversion, see the
# module notes), refactorizes reusing the symbolic analysis, and solves sals.rhs into ls.h_vec.
function cudss_solve!(ls::CUDSSDirectSolver)
    nzval, host_vals, perm = ls.sals.M.nzval, ls.host_vals, ls.vals_perm
    @inbounds for k in eachindex(host_vals) # CPU-side gather into the GPU matrix's own nonzero order, into a preallocated buffer
        host_vals[k] = nzval[perm[k]]
    end
    copyto!(ls.A_gpu.nzVal, host_vals) # host -> device, straight into the existing matrix's values
    ls.general ? lu!(ls.solver, ls.A_gpu) : cholesky!(ls.solver, ls.A_gpu) # refactorize in place, reusing the symbolic analysis
    copyto!(ls.b_gpu, ls.sals.rhs)
    ls.x_gpu .= ls.solver \ ls.b_gpu
    copyto!(ls.h_vec, ls.x_gpu)
    return ls.h_vec
end
