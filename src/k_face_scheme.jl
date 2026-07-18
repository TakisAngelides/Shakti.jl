abstract type AbstractKFaceScheme end

struct Arithmetic <: AbstractKFaceScheme end
struct Harmonic <: AbstractKFaceScheme end

@inline compute_K_face(::Arithmetic, K, i1, j1, i2, j2) = (K[i1, j1] + K[i2, j2]) / 2
@inline compute_K_face(::Harmonic, K, i1, j1, i2, j2) = (2 * K[i1, j1] * K[i2, j2]) / (K[i1, j1] + K[i2, j2] + eps(eltype(K)))
