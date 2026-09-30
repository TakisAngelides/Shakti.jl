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
@inline compute_K_face(kfs::AbstractKFaceScheme, K, i1, j1, i2, j2) = face_mean(kfs, K[i1, j1], K[i2, j2])

"""
$(TYPEDSIGNATURES)

Scalar form of [`compute_K_face`](@ref): the arithmetic or harmonic mean of two cell values.
"""
@inline face_mean(::Arithmetic, a, b) = (a + b) / 2
@inline face_mean(::Harmonic, a, b) = (2 * a * b) / (a + b + eps(typeof(a)))

"""
$(TYPEDSIGNATURES)

`true` for the two Dirichlet drainage-boundary mask values (`OCEAN`/`LAND`).
"""
@inline is_dirichlet(m) = (m == OCEAN) | (m == LAND)

"""
$(TYPEDSIGNATURES)

Face value of a cell-centred conductance (`Ka` in a cell with mask `ma`, `Kb` in its neighbour
with mask `mb`), symmetric in its two cells so both sides of a face read the same value -- the
face conductance [`compute_face_flux!`](@ref) (`water_flux.jl`) builds `s.K_x`/`s.K_y` from, and
therefore exactly what the linear system assembles:

- both `GROUNDED`: the K-face scheme's mean (`kfs`);
- one `GROUNDED`, the other `OCEAN`/`LAND`: the grounded cell's own value (see
  [`boundary_K_face`](@ref) for why the Dirichlet cell's placeholder `0` must not be averaged in);
- anything else (a face touching `OTHER_BASIN`/`FROZEN_BED`, or between two non-solved cells): `0`,
  a zero-flux face.
"""
@inline function face_conductance(kfs::AbstractKFaceScheme, Ka, ma, Kb, mb)
    ga, gb = ma == GROUNDED, mb == GROUNDED
    if ga & gb
        return face_mean(kfs, Ka, Kb)
    elseif ga & is_dirichlet(mb)
        return Ka
    elseif gb & is_dirichlet(ma)
        return Kb
    else
        return zero(Ka)
    end
end

"""
$(TYPEDSIGNATURES)

Face conductance between a solved (`GROUNDED`) cell `(i1,j1)` and its neighbour `(i2,j2)`, aware
of what kind of cell the neighbour is. The assembly kernels no longer call this -- they read the
face transmissivities `s.K_x`/`s.K_y` built by [`compute_face_flux!`](@ref) via
[`face_conductance`](@ref), which applies the same rules below -- it is kept for external use.

# Notes

- `OTHER_BASIN`/`FROZEN_BED`: unsolved/frozen -- zero-flux (Neumann) face. `FROZEN_BED`'s own `K`
  is genuinely `0` there (`b=0`, no gap), so this isn't just a convention matching `OTHER_BASIN`'s
  -- either scheme would already give `0` on that face from `K[i2,j2]=0` alone (harmonic
  trivially, arithmetic because `boundary_K_face` never blends toward a real `K` for a
  non-`GROUNDED` neighbour in the first place); the explicit branch keeps the *reason* (no water
  crosses into a frozen cell) stated directly rather than left to fall out of the arithmetic.
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
    if m2 == OTHER_BASIN || m2 == FROZEN_BED
        return zero(eltype(K))
    elseif m2 == OCEAN || m2 == LAND
        return K[i1, j1] # if the neighbour is land or ocean, there is no meaningful conductivity value K there so we just use the center value at i1, j1 for that cell face
    else
        return compute_K_face(kfs, K, i1, j1, i2, j2)
    end
end
