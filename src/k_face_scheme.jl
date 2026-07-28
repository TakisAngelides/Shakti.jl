"""
$(TYPEDSIGNATURES)

How hydraulic conductivity `K` is combined across two neighbouring `GROUNDED` cells' shared face
into a single face conductance -- multiple dispatch on the concrete subtype picks arithmetic vs.
harmonic averaging (see [`compute_K_face`](@ref) and [`boundary_K_face`](@ref)).
"""
abstract type AbstractKFaceScheme end

"""
$(TYPEDSIGNATURES)

Arithmetic-mean face conductance.
"""
struct Arithmetic <: AbstractKFaceScheme end

"""
$(TYPEDSIGNATURES)

Harmonic-mean face conductance: penalizes a face where either neighbour has low conductivity more
strongly than the arithmetic mean does.
"""
struct Harmonic <: AbstractKFaceScheme end

"""
$(TYPEDSIGNATURES)

Face conductance between cells `(i1,j1)` and `(i2,j2)`, `(K[i1,j1] + K[i2,j2]) / 2` under
[`Arithmetic`](@ref) or `2*K[i1,j1]*K[i2,j2] / (K[i1,j1] + K[i2,j2] + eps)` under [`Harmonic`](@ref).
Here @inline can help the compiler optimize its code and reduce the cost of function calls.
"""
@inline compute_K_face(::Arithmetic, K, i1, j1, i2, j2) = (K[i1, j1] + K[i2, j2]) / 2
@inline compute_K_face(::Harmonic, K, i1, j1, i2, j2) = (2 * K[i1, j1] * K[i2, j2]) / (K[i1, j1] + K[i2, j2] + eps(eltype(K)))

"""
$(TYPEDSIGNATURES)

Face conductance between a solved (`GROUNDED`) cell `(i1,j1)` and its neighbour `(i2,j2)`, aware
of what kind of cell the neighbour is.

# Notes

- `OTHER_BASIN`: unsolved/frozen -- zero-flux (Neumann) face.
- `OCEAN`/`LAND`: real Dirichlet drainage boundaries, but `K` there is a bookkeeping placeholder
  (`b` is forced to `0` at those cells in [`set_initial_conditions!`](@ref), since they have no
  physical gap height), not an actual conductivity. Folding that `0` into [`compute_K_face`](@ref)
  would spuriously choke off drainage -- especially under [`Harmonic`](@ref), where one side being
  `0` collapses the whole face to `0`. Uses the solved cell's own `K` instead, i.e. treats
  conductivity as extending unchanged up to the boundary.
- `GROUNDED`: both sides are real hydrology cells, use the K-face scheme (`kfs`).
"""
@inline function boundary_K_face(kfs::AbstractKFaceScheme, K, mask, i1, j1, i2, j2)
    m2 = mask[i2, j2]
    if m2 == OTHER_BASIN
        return zero(eltype(K))
    elseif m2 == OCEAN || m2 == LAND
        return K[i1, j1] # if the neighbour is land or ocean, there is no meaningful conductivity value K there so we just use the center value at i1, j1 for that cell face
    else
        return compute_K_face(kfs, K, i1, j1, i2, j2)
    end
end
