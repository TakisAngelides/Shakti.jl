# Scalar-returning CUBLAS reductions (dot, nrm2) with the device result buffer kept alive until it has been read.
#
# CUDA.jl 5.11 returns the result of CUBLAS.dot and CUBLAS.nrm2 through a CuRef, and `getindex(::CuRefValue)`
# (src/refpointer.jl) preserves only its host-side Ref while copying the value back: between `unsafe_convert` of the
# device pointer and the `unsafe_copyto!`, nothing roots the CuRefValue, so a garbage collection triggered on another
# thread can free its device buffer and the copy fails with "invalid argument" in cuMemcpyDtoHAsync. It shows up in
# long GPU runs under many threads with heavy host allocation, a few times per million dots (reproduced and traced in
# PostDoc_Latex_Notes/Notes_on_Shakti/cuda_dot_curef_gc_race.tex). The functions below make the same CUBLAS calls
# and hold the CuRef with `GC.@preserve` across the read. Krylov reaches dot and norm through `kdot` and `knorm`
# (`kdotr` and `knorm_elliptic` call these), so more specific CuVector methods of the two route every CG and BiCGStab
# reduction here; Shakti's own GPU dots (estimate_eigenvalue_bounds!) go through `safe_dot`. Remove this file once
# CUDA.jl preserves the CuRefValue in `getindex` itself.

const CublasReal = Union{Float32, Float64}

function preserved_dot(n::Integer, x::CuVector{T}, y::CuVector{T}) where {T <: CublasReal}
    result = CuRef{T}()
    GC.@preserve result begin
        CUDA.CUBLAS.dot(n, x, y, result)
        return result[]
    end
end

function preserved_nrm2(n::Integer, x::CuVector{T}) where {T <: CublasReal}
    result = CuRef{T}()
    GC.@preserve result begin
        CUDA.CUBLAS.nrm2(n, x, result)
        return result[]
    end
end

Krylov.kdot(n::Integer, x::CuVector{T}, y::CuVector{T}) where {T <: CublasReal} = preserved_dot(n, x, y)
Krylov.knorm(n::Integer, x::CuVector{T}) where {T <: CublasReal} = preserved_nrm2(n, x)

safe_dot(x::CuVector{T}, y::CuVector{T}) where {T <: CublasReal} = preserved_dot(length(x), x, y)
