abstract type AbstractKFaceScheme end

struct Arithmetic <: AbstractKFaceScheme end
struct Harmonic <: AbstractKFaceScheme end

@inline compute_K_face(::Arithmetic, K, i1, j1, i2, j2) = (K[i1, j1] + K[i2, j2]) / 2
@inline compute_K_face(::Harmonic, K, i1, j1, i2, j2) = (2 * K[i1, j1] * K[i2, j2]) / (K[i1, j1] + K[i2, j2] + eps(eltype(K)))

# Face conductance between a solved (GROUNDED) cell (i1,j1) and its neighbour
# (i2,j2), aware of what kind of cell the neighbour is:
# - OTHER_BASIN: unsolved/frozen -- zero-flux (Neumann) face.
# - OCEAN/LAND: real Dirichlet drainage boundaries, but K there is a
#   bookkeeping placeholder (b is forced to 0 at those cells in
#   initial_conditions.jl, since they have no physical gap height), not an
#   actual conductivity. Folding that 0 into compute_K_face would spuriously
#   choke off drainage -- especially under Harmonic, where one side being 0
#   collapses the whole face to 0. Use the solved cell's own K instead, i.e.
#   treat conductivity as extending unchanged up to the boundary.
# - GROUNDED: both sides are real hydrology cells, use the K-face scheme.
@inline function boundary_K_face(kfs::AbstractKFaceScheme, K, mask, i1, j1, i2, j2)
    m2 = mask[i2, j2]
    if m2 == OTHER_BASIN
        return zero(eltype(K))
    elseif m2 == OCEAN || m2 == LAND
        return K[i1, j1]
    else
        return compute_K_face(kfs, K, i1, j1, i2, j2)
    end
end
