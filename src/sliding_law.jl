"""
$(TYPEDSIGNATURES)

How `taub` (basal shear stress, feeding [`compute_mdot!`](@ref)'s frictional-heating term) is
obtained -- multiple dispatch on the concrete subtype, decided once per `Simulation` (same idiom as
[`MeltTerms`](@ref), `melt_rate.jl`). Every law except [`PrescribedSlidingLaw`](@ref) gives the
stress as `taub = T(N, |u_b|) * u_b/|u_b|`, a magnitude `T` along the sliding direction,
recomputed from `N`/`ub` every Picard iteration; see the tutorial's sliding-law table for the
formulas and references.

# Notes

The original SHAKTI paper (Sommers et al. 2018) never actually specifies this -- `taub` is listed
in its Table 1 as a state variable with no formula, since it normally comes from a coupled
ice-dynamics solve; Table 2's constants have no friction coefficient at all. So this is a real
modeling choice, not just an internal solver detail; keeping it explicit and swappable rather
than silently baking in one assumption is the point of this abstraction.

Adding a law takes a struct, a `kernel_args` method returning its isbits parameters and
its face coefficients, and a `basal_stress` method for `T` -- the kernels below are shared.
"""
abstract type AbstractSlidingLaw end

"""
$(TYPEDSIGNATURES)

`taub_x`/`taub_y` prescribed once ([`initialize_taub!`](@ref), from whatever formula or data the
caller supplies -- e.g. a driving-stress balance `taub = rho_i*g*H*(surface slope)`, the standard
parameter-free choice for synthetic slab/margin setups) and left unchanged thereafter:
[`compute_taub_x!`](@ref) etc. are no-ops for this law.
"""
struct PrescribedSlidingLaw <: AbstractSlidingLaw end

"""
$(TYPEDSIGNATURES)

Stages a cell-centered drag/friction coefficient `C` (an `(nx, ny)` array, e.g. an inverted
per-cell field, or a uniform scalar) onto x-/y-faces: boundary faces take the value of the nearest
cell center, interior faces the arithmetic mean of their two neighbours -- same convention as
[`compute_b_x!`](@ref)/[`compute_b_y!`](@ref) (`gap_height.jl`). Shared by every sliding law that
stores its own coefficients staggered onto faces once at construction rather than in the
per-iteration hot path. Returns plain host `F`-typed arrays `(Cx, Cy)`; callers apply their own
`Data.Array` conversion (and squaring, for [`LinearSlidingLaw`](@ref)) on top.
"""
function stagger_to_faces(g::Grid, C)
    nx, ny = g.nx, g.ny
    F = eltype(g.x)
    C_cc = C isa AbstractArray ? F.(C) : fill(F(C), nx, ny) # promote a scalar to a uniform field so the staggering below is one code path either way

    # Edge faces take the value of the center, otherwise we use the arithmetic mean of the two centers next to a face
    Cx = zeros(F, nx + 1, ny)
    Cx[1, :]    .= C_cc[1, :]
    Cx[nx+1, :] .= C_cc[nx, :]
    Cx[2:nx, :] .= (C_cc[1:nx-1, :] .+ C_cc[2:nx, :]) ./ 2

    Cy = zeros(F, nx, ny + 1)
    Cy[:, 1]    .= C_cc[:, 1]
    Cy[:, ny+1] .= C_cc[:, ny]
    Cy[:, 2:ny] .= (C_cc[:, 1:ny-1] .+ C_cc[:, 2:ny]) ./ 2

    return Cx, Cy
end

# stagger_to_faces as device arrays, for the laws below
device_faces(g::Grid, C) = map(Data.Array, stagger_to_faces(g, C))

# A coefficient that is the same on every face, indexed like the staggered arrays inside the kernels. isbits, so it
# passes to a GPU kernel as it is.
struct UniformField{F}
    v::F
end
@inline Base.getindex(u::UniformField, i, j) = u.v

# The isbits part of a law that the kernels see: its scalar parameters `p` (a NamedTuple), tagged with the law's name
# `L` so `basal_stress` dispatches on it. The law structs themselves hold device arrays, which a GPU kernel can't take.
struct LawRule{L, P}
    p::P
end
LawRule{L}(p::P) where {L, P} = LawRule{L, P}(p)

# ---------------------------------------------------------------------------------------------------------------------
# The laws. Each one's `basal_stress(rule, N, v, lambda, coefficients)` is the magnitude T >= 0 for the face effective
# pressure N >= 0 (see bed_N), the face sliding speed v = |u_b| > 0, the face lambda (Shakti's own N^n*lambda velocity
# scale, read by the first two laws only) and the tuple of the law's face coefficients at that face.
# ---------------------------------------------------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Regularized-Coulomb sliding law: `taub -> C*N` as `ub/N^n*lambda -> infinity` (hard Coulomb
limit), but stays finite (unlike a bare Coulomb law) as `ub -> 0`. Recomputed from `N`/`ub` every
Picard iteration (see [`compute_taub_x!`](@ref) etc. below).

# Notes

`n`/`inv_n` are read from `p.n_exp`/`p.inv_n_exp`, canonicalized once at `ModelParameters`
construction (see `model_parameters.jl`'s `pow`/`canonical_exponent` note), not per call or per
grid cell.
"""
struct RegularizedCoulombSlidingLaw{F <: AbstractFloat} <: AbstractSlidingLaw
    C::F # Coulomb friction coefficient
end

"""
$(TYPEDSIGNATURES)

Regularized-Coulomb sliding law with a spatially-varying friction coefficient `C(x, y)` -- e.g. an
inverted field like Kazmierczak et al. (2024)'s Thwaites Glacier `C`, derived from their `beta2`/
`Neff`/`ub` output fields via `C = beta2*ub / (Neff*(ub/(ub+v0))^(1/m))` (their Eq. 1 solved for
`C`) -- rather than [`RegularizedCoulombSlidingLaw`](@ref)'s uniform scalar. Same `taub -> C*N`
functional form and Picard-loop recomputation as that law; `C` is given at cell centers and
staggered onto faces once at construction (see
[`RegularizedCoulombFieldSlidingLaw(::Grid, C)`](@ref)) rather than in the per-iteration hot path.
"""
struct RegularizedCoulombFieldSlidingLaw{A <: AbstractArray} <: AbstractSlidingLaw
    Cx::A # C staggered onto x-faces (Nx+1, Ny)
    Cy::A # C staggered onto y-faces (Nx, Ny+1)
end

"""
$(TYPEDSIGNATURES)

Builds a [`RegularizedCoulombFieldSlidingLaw`](@ref) on grid `g` from a friction coefficient `C`
given at cell centers (an `(nx, ny)` array, e.g. an inverted per-cell field) or as a uniform
scalar, staggering it onto faces once here rather than in the per-iteration hot path.
"""
RegularizedCoulombFieldSlidingLaw(g::Grid, C) = RegularizedCoulombFieldSlidingLaw(device_faces(g, C)...)

kernel_args(sl::RegularizedCoulombSlidingLaw, p) = (LawRule{:RegularizedCoulomb}((n = p.n_exp, inv_n = p.inv_n_exp)), (UniformField(sl.C),), (UniformField(sl.C),))
kernel_args(sl::RegularizedCoulombFieldSlidingLaw, p) = (LawRule{:RegularizedCoulomb}((n = p.n_exp, inv_n = p.inv_n_exp)), (sl.Cx,), (sl.Cy,))
@inline basal_stress(r::LawRule{:RegularizedCoulomb}, N, v, lambda, (C,)) = C * N * pow(v / (v + pow(N, r.p.n) * lambda), r.p.inv_n)

"""
$(TYPEDSIGNATURES)

Regularized-Coulomb sliding law using Kazmierczak et al. (2024)/Joughin et al. (2019)'s Eq. 1
literally: `taub = C*N*(||vb||/(||vb||+v0))^q * vb/||vb||`, with `v0` a fixed constant velocity
threshold (their Appendix A: `v0 = 300 m/a = 9.51e-6 m/s`), NOT [`RegularizedCoulombSlidingLaw`](@ref)/
[`RegularizedCoulombFieldSlidingLaw`](@ref)'s `N^n*lambda` velocity scale (Shakti's own, inherited
from the original SHAKTI/Sommers et al. 2018 cavity-opening physics -- see those types' docstrings).
Use this one when the goal is reproducing the paper's own regularization, not Shakti's native one.
With `C = tan(phi)` and `q = 1/p` it is also Zoet and Iverson (2020)'s slip law for till beds.

# Notes

`v0` must be in the same units as `ub_x`/`ub_y`/`abs_ub` (Shakti's internal m/s, not the paper's
m/a -- convert before constructing). The exponent `q` (`1/m` in K24, Yelmo's `beta_q`) is a field of the law,
independent of Glen's `n` (default `1/3`, the old `1/n` at `n = 3`); `p.n_exp`/`p.inv_n_exp` are not read by this law.
"""
struct RegularizedCoulombV0SlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    Cx::A # C staggered onto x-faces (Nx+1, Ny)
    Cy::A # C staggered onto y-faces (Nx, Ny+1)
    v0::F # constant velocity threshold, same units as ub_x/ub_y (m/s)
    q::F  # velocity exponent (1/m in K24; Yelmo's beta_q), independent of Glen's n. Default 1/3 == old p.inv_n_exp at n=3
end

"""
$(TYPEDSIGNATURES)

Builds a [`RegularizedCoulombV0SlidingLaw`](@ref) on grid `g` from a friction coefficient `C` given
at cell centers (an `(nx, ny)` array, e.g. an inverted per-cell field) or as a uniform scalar, and a
constant velocity threshold `v0` (m/s).
"""
function RegularizedCoulombV0SlidingLaw(g::Grid, C, v0; q = 1/3)
    F = eltype(g.x)
    return RegularizedCoulombV0SlidingLaw(device_faces(g, C)..., F(v0), F(q))
end

kernel_args(sl::RegularizedCoulombV0SlidingLaw, p) = (LawRule{:RegularizedCoulombV0}((v0 = sl.v0, q = sl.q)), (sl.Cx,), (sl.Cy,))
@inline basal_stress(r::LawRule{:RegularizedCoulombV0}, N, v, lambda, (C,)) = C * N * pow(v / (v + r.p.v0), r.p.q)

# Linear (viscous) sliding law: taub = C^2*N*u_b (Sommers and others 2023,
# Table 1). Written there as a scalar equation taub = C^2*N*|u_b| -- meant
# as "magnitude, direction opposite to u_b" -- but the |u_b| in that
# magnitude and the 1/|u_b| from normalizing direction cancel exactly,
# leaving a plain vector equation linear in u_b with no magnitude/direction
# split (and no ub->0 guard) needed at all: its basal_shear below is
# C^2*N*u itself.
"""
$(TYPEDSIGNATURES)

Linear (viscous) sliding law `taub = C^2*N*u_b` (Sommers et al. 2023, Table 1; ISSM's Shakti friction
law). `C` is given at cell centers or as a uniform scalar; see [`LinearSlidingLaw(::Grid, C)`](@ref).
"""
struct LinearSlidingLaw{A <: AbstractArray} <: AbstractSlidingLaw
    Cx2::A # C^2 staggered onto x-faces (Nx+1, Ny)
    Cy2::A # C^2 staggered onto y-faces (Nx, Ny+1)
end

"""
$(TYPEDSIGNATURES)

Builds a [`LinearSlidingLaw`](@ref) on grid `g` from a drag coefficient `C` given at cell centers
(an `(nx, ny)` array, e.g. an inverted per-cell field) or as a uniform scalar, staggering it onto
faces and squaring it once here rather than in the per-iteration hot path.
"""
function LinearSlidingLaw(g::Grid, C)
    Cx, Cy = stagger_to_faces(g, C)
    return LinearSlidingLaw(Data.Array(Cx .^ 2), Data.Array(Cy .^ 2))
end

kernel_args(sl::LinearSlidingLaw, p) = (LawRule{:Linear}(()), (sl.Cx2,), (sl.Cy2,))

"""
$(TYPEDSIGNATURES)

Weertman (1957) power law `taub = C*|u_b|^m * u_b/|u_b|`, with no effective-pressure dependence: the
stress is the same on a flooded and a dry bed. `m = 1` is the linear viscous law `taub = C*u_b`.
`C` [Pa (s/m)^m] at cell centers or a uniform scalar; `ub` in m/s.
"""
struct WeertmanSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    Cx::A
    Cy::A
    m::F # velocity exponent
end
WeertmanSlidingLaw(g::Grid, C; m = 1/3) = WeertmanSlidingLaw(device_faces(g, C)..., eltype(g.x)(m))

kernel_args(sl::WeertmanSlidingLaw, p) = (LawRule{:Weertman}((m = sl.m,)), (sl.Cx,), (sl.Cy,))
@inline basal_stress(r::LawRule{:Weertman}, N, v, lambda, (C,)) = C * pow(v, r.p.m)

"""
$(TYPEDSIGNATURES)

Budd et al. (1979) power law `taub = C*N^q*|u_b|^m * u_b/|u_b|`, the general effective-pressure
dependent power law. ISSM's Budd law `C^2 N^r |u_b|^s` is `C -> C^2, q = r, m = s`; Boulton and
Hindmarsh (1987)'s till rheology `u_b = A*tau^a*N^-b` is `C = A^(-1/a), m = 1/a, q = b/a`; `q = 0`
is [`WeertmanSlidingLaw`](@ref), `q = m = 1` with `C -> C^2` is [`LinearSlidingLaw`](@ref).
"""
struct BuddSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    Cx::A
    Cy::A
    m::F # velocity exponent
    q::F # effective-pressure exponent
end
BuddSlidingLaw(g::Grid, C; m = 1/3, q = 1) = BuddSlidingLaw(device_faces(g, C)..., eltype(g.x)(m), eltype(g.x)(q))

kernel_args(sl::BuddSlidingLaw, p) = (LawRule{:Budd}((m = sl.m, q = sl.q)), (sl.Cx,), (sl.Cy,))
@inline basal_stress(r::LawRule{:Budd}, N, v, lambda, (C,)) = C * pow(N, r.p.q) * pow(v, r.p.m)

"""
$(TYPEDSIGNATURES)

Coulomb (plastic) friction `taub = mu*N * u_b/|u_b|`: the bed is at its yield stress wherever it
slides. `mu` is Iken (1981)'s bound `tan(beta)` for a hard bed, or `tan(phi)` (Mohr-Coulomb, `phi` the
till friction angle) for a till bed. Since `ub` is given, the stress is defined, unlike in an
ice-flow model where the yield condition alone does not fix the speed.
"""
struct CoulombSlidingLaw{A <: AbstractArray} <: AbstractSlidingLaw
    mux::A
    muy::A
end
CoulombSlidingLaw(g::Grid, mu) = CoulombSlidingLaw(device_faces(g, mu)...)

kernel_args(sl::CoulombSlidingLaw, p) = (LawRule{:Coulomb}(()), (sl.mux,), (sl.muy,))
@inline basal_stress(::LawRule{:Coulomb}, N, v, lambda, (mu,)) = mu * N

"""
$(TYPEDSIGNATURES)

Tsai et al. (2015): the smaller of a power law and a Coulomb stress,
`taub = min(C*N^q*|u_b|^m, f*N) * u_b/|u_b|`. Tsai's own law has `q = 0`; `q = 1` with `C -> C^2`,
`f -> C_c^2` is ISSM's Coulomb-limited Budd law.
"""
struct TsaiSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    Cx::A
    Cy::A
    fx::A # Coulomb coefficient on the faces
    fy::A
    m::F
    q::F
end
function TsaiSlidingLaw(g::Grid, C, f; m = 1/3, q = 0)
    F = eltype(g.x)
    return TsaiSlidingLaw(device_faces(g, C)..., device_faces(g, f)..., F(m), F(q))
end

kernel_args(sl::TsaiSlidingLaw, p) = (LawRule{:Tsai}((m = sl.m, q = sl.q)), (sl.Cx, sl.fx), (sl.Cy, sl.fy))
@inline basal_stress(r::LawRule{:Tsai}, N, v, lambda, (C, f)) = min(C * pow(N, r.p.q) * pow(v, r.p.m), f * N)

"""
$(TYPEDSIGNATURES)

Schoof (2005)'s cavitation law in ISSM's form,
`taub = C^2*|u_b|^m / (1 + (C^2/(Cmax*N))^(1/m) * |u_b|)^m * u_b/|u_b|`: a Weertman law `C^2 |u_b|^m`
at low speed, the Coulomb stress `Cmax*N` (Iken's bound) at high speed. Cornford et al. (2020)'s
MISMIP+ law `C alpha^2 N u^m / (C^(1/m) u + (alpha^2 N)^(1/m))^m` is `C -> sqrt(C)`, `Cmax = alpha^2`;
McArthur et al. (2026)'s hydrology-modified Schoof law, transition parameter `zeta`, is
`Cmax = C^2/zeta`.
"""
struct SchoofSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    Cx::A
    Cy::A
    Cmaxx::A
    Cmaxy::A
    m::F
end
SchoofSlidingLaw(g::Grid, C, Cmax; m = 1/3) = SchoofSlidingLaw(device_faces(g, C)..., device_faces(g, Cmax)..., eltype(g.x)(m))

kernel_args(sl::SchoofSlidingLaw, p) = (LawRule{:Schoof}((m = sl.m, inv_m = 1 / sl.m)), (sl.Cx, sl.Cmaxx), (sl.Cy, sl.Cmaxy))
@inline function basal_stress(r::LawRule{:Schoof}, N, v, lambda, (C, Cmax))
    N > 0 || return zero(N) # the Coulomb limit Cmax*N
    C2 = C * C
    return C2 * pow(v, r.p.m) / pow(1 + pow(C2 / (Cmax * N), r.p.inv_m) * v, r.p.m)
end

"""
$(TYPEDSIGNATURES)

Gagliardini et al. (2007)'s cavitation law (ISSM's hydro friction, Elmer/Ice's), from cavities over
a sinusoidal bed: `taub = C*N * (chi/(1 + alpha*chi^q))^(1/n) * u_b/|u_b|` with
`chi = |u_b|/(C^n N^n A_s)` and `alpha = (q-1)^(q-1)/q^q`. `C` is the maximum bed slope (`taub <= C*N`),
`A_s` [m Pa^-n s^-1] the sliding parameter without cavities, `q >= 1` the post-peak exponent: `q > 1`
softens past the peak, `q = 1` is a regularized Coulomb law that never softens.
"""
struct GagliardiniSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    Cx::A
    Cy::A
    As::F
    q::F
    n::F
end
GagliardiniSlidingLaw(g::Grid, C, As; q = 1, n = 3) = GagliardiniSlidingLaw(device_faces(g, C)..., eltype(g.x).((As, q, n))...)

function kernel_args(sl::GagliardiniSlidingLaw, p)
    alpha = (sl.q - 1)^(sl.q - 1) / sl.q^sl.q # 1 at q = 1 (0^0 = 1)
    return (LawRule{:Gagliardini}((As = sl.As, q = sl.q, n = sl.n, inv_n = 1 / sl.n, alpha = alpha)), (sl.Cx,), (sl.Cy,))
end
@inline function basal_stress(r::LawRule{:Gagliardini}, N, v, lambda, (C,))
    N > 0 || return zero(N)
    chi = v / (pow(C * N, r.p.n) * r.p.As)
    return C * N * pow(chi / (1 + r.p.alpha * pow(chi, r.p.q)), r.p.inv_n)
end

"""
$(TYPEDSIGNATURES)

Regularized Coulomb law with an effective-pressure dependent transition speed `(K*N)^m`,
`taub = C*N * (|u_b|/(|u_b| + (K*N)^m))^(1/m) * u_b/|u_b|` (Helanow et al. 2021, from 3-D cavity models
over measured hard beds; ISSM's regcoulomb2). It tends to `C*N` at high speed. Same form as [`GagliardiniSlidingLaw`](@ref) at `q = 1` with `K^m = A_s C^n`.
"""
struct HelanowSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    Cx::A
    Cy::A
    Kx::A # [(m/s)^(1/m) / Pa]
    Ky::A
    m::F
end
HelanowSlidingLaw(g::Grid, C, K; m = 3) = HelanowSlidingLaw(device_faces(g, C)..., device_faces(g, K)..., eltype(g.x)(m))

kernel_args(sl::HelanowSlidingLaw, p) = (LawRule{:Helanow}((m = sl.m, inv_m = 1 / sl.m)), (sl.Cx, sl.Kx), (sl.Cy, sl.Ky))
@inline basal_stress(r::LawRule{:Helanow}, N, v, lambda, (C, K)) = C * N * pow(v / (v + pow(K * N, r.p.m)), r.p.inv_m)

"""
$(TYPEDSIGNATURES)

PISM's pseudo-plastic law (Bueler and Brown 2009), `taub = tau_c * (|u_b|/u0)^q * u_b/|u_b|` with the
Mohr-Coulomb yield stress `tau_c = c0 + tan(phi)*N`. `q = 0` is Coulomb, `q = 1` linear;
`u0` (m/s) is the threshold speed at which `taub = tau_c`, PISM's default 100 m/a.
"""
struct PseudoPlasticSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    tanphix::A
    tanphiy::A
    q::F
    u0::F
    c0::F # till cohesion [Pa]
end
PseudoPlasticSlidingLaw(g::Grid, tanphi; q = 0.25, u0 = 100 / 31_557_600, c0 = 0) = PseudoPlasticSlidingLaw(device_faces(g, tanphi)..., eltype(g.x).((q, u0, c0))...)

kernel_args(sl::PseudoPlasticSlidingLaw, p) = (LawRule{:PseudoPlastic}((q = sl.q, u0 = sl.u0, c0 = sl.c0)), (sl.tanphix,), (sl.tanphiy,))
@inline basal_stress(r::LawRule{:PseudoPlastic}, N, v, lambda, (tanphi,)) = (r.p.c0 + tanphi * N) * pow(v / r.p.u0, r.p.q)

"""
$(TYPEDSIGNATURES)

Weertman (1957)/Nye (1969)/Kamb (1970)'s sliding by regelation plus enhanced creep around bed
obstacles: the two speeds add, `|u_b| = A_R*taub + A_C*taub^n`, which is solved for `taub` by Newton's
method on every face. Independent of `N`, like [`WeertmanSlidingLaw`](@ref). `A_R` [m s^-1 Pa^-1] and
`A_C` [m s^-1 Pa^-n] at cell centers or uniform scalars.
"""
struct RegelationCreepSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    ARx::A
    ARy::A
    ACx::A
    ACy::A
    n::F
end
RegelationCreepSlidingLaw(g::Grid, A_R, A_C; n = 3) = RegelationCreepSlidingLaw(device_faces(g, A_R)..., device_faces(g, A_C)..., eltype(g.x)(n))

kernel_args(sl::RegelationCreepSlidingLaw, p) = (LawRule{:RegelationCreep}((n = sl.n, inv_n = 1 / sl.n)), (sl.ARx, sl.ACx), (sl.ARy, sl.ACy))
@inline function basal_stress(r::LawRule{:RegelationCreep}, N, v, lambda, (AR, AC))
    n = r.p.n
    # Each mechanism alone carries the whole speed at a larger stress, so the smaller of the two is an upper bound of the
    # root, and f(tau) = AR*tau + AC*tau^n - v is increasing and convex: Newton from above falls monotonically onto it.
    tau = min(v / AR, pow(v / AC, r.p.inv_n))
    for _ in 1:50
        tau_new = tau - (AR * tau + AC * pow(tau, n) - v) / (AR + n * AC * pow(tau, n - 1))
        tau_new < tau || break # converged to round-off
        tau = tau_new
    end
    return tau
end

"""
$(TYPEDSIGNATURES)

McArthur et al. (2026)'s linear combination of a Coulomb and a Weertman stress,
`taub = (Cmax*N*xi + C^2*|u_b|^m) / (1 + xi) * u_b/|u_b|` with `xi = alpha*|u_b|/N^n`: the weights
`xi/(1+xi)` and `1/(1+xi)` sum to one, so `Cmax` and `C` are set independently, unlike in
[`SchoofSlidingLaw`](@ref).
"""
struct LinearCombinationSlidingLaw{A <: AbstractArray, F <: AbstractFloat} <: AbstractSlidingLaw
    Cx::A
    Cy::A
    Cmaxx::A
    Cmaxy::A
    alpha::F # [Pa^n s/m]
    m::F
    n::F
end
LinearCombinationSlidingLaw(g::Grid, C, Cmax, alpha; m = 1/3, n = 3) = LinearCombinationSlidingLaw(device_faces(g, C)..., device_faces(g, Cmax)..., eltype(g.x).((alpha, m, n))...)

kernel_args(sl::LinearCombinationSlidingLaw, p) = (LawRule{:LinearCombination}((alpha = sl.alpha, m = sl.m, n = sl.n)), (sl.Cx, sl.Cmaxx), (sl.Cy, sl.Cmaxy))
@inline function basal_stress(r::LawRule{:LinearCombination}, N, v, lambda, (C, Cmax))
    N > 0 || return zero(N) # xi -> infinity: the Coulomb stress Cmax*N
    xi = r.p.alpha * v / pow(N, r.p.n)
    return (Cmax * N * xi + C * C * pow(v, r.p.m)) / (1 + xi)
end

"""
$(TYPEDSIGNATURES)

Any law of the form `taub = f(N, |u_b|, C) * u_b/|u_b|`, for the families with no single closed form
(Lliboutry 1968, Fowler 1987) or a law of one's own. `f` must return the magnitude (`>= 0`) and,
to run on a GPU, be a plain function or a closure over isbits values. `N >= 0` and `|u_b| > 0` when `f`
is called; `C` is the face value of the cell field (or scalar) passed here.
"""
struct FunctionSlidingLaw{Fn, A <: AbstractArray} <: AbstractSlidingLaw
    f::Fn
    Cx::A
    Cy::A
end
FunctionSlidingLaw(g::Grid, f, C = 1) = FunctionSlidingLaw(f, device_faces(g, C)...)

kernel_args(sl::FunctionSlidingLaw, p) = (LawRule{:Function}((f = sl.f,)), (sl.Cx,), (sl.Cy,))
@inline basal_stress(r::LawRule{:Function}, N, v, lambda, (C,)) = r.p.f(N, v, C)

# ---------------------------------------------------------------------------------------------------------------------
# Shared kernels
# ---------------------------------------------------------------------------------------------------------------------

# The face component of taub: T times the direction cosine u/|u_b|, and 0 where the bed doesn't slide (the 0/0 of both
# components vanishing). T*(u/|u_b|), not (T/|u_b|)*u: T/|u_b| overflows when a run goes astray (N ~ 1e300 in a gap
# runaway), where the direction cosine is at most 1. The linear law is linear straight through u_b = 0, so it is
# C^2*N*u with no guard.
@inline basal_shear(r::LawRule, N, v, lambda, c, u) = v > 0 ? basal_stress(r, N, v, lambda, c) * (u / v) : zero(N)
@inline basal_shear(::LawRule{:Linear}, N, v, lambda, (C2,), u) = C2 * N * u

# Face value of a cell-centred field for the basal stress on that face: the mean of the two cells,
# except on a face between a GROUNDED cell and any other cell (OCEAN, LAND, OTHER_BASIN, FROZEN_BED),
# where it is the GROUNDED cell's own value. The face lies on that cell's bed, and the other side's N
# (0 off the hydrology, or a frozen bed's po) says nothing about it.
@inline function face_x(F, mask, ix, iy)
    ga, gb = mask[ix, iy] == GROUNDED, mask[ix-1, iy] == GROUNDED
    return ga == gb ? (F[ix, iy] + F[ix-1, iy]) / 2 : (ga ? F[ix, iy] : F[ix-1, iy])
end
@inline function face_y(F, mask, ix, iy)
    ga, gb = mask[ix, iy] == GROUNDED, mask[ix, iy-1] == GROUNDED
    return ga == gb ? (F[ix, iy] + F[ix, iy-1]) / 2 : (ga ? F[ix, iy] : F[ix, iy-1])
end

# face_x/face_y on every face, the outer boundary faces taking their one cell's value
@inline edge_face_x(F, mask, ix, iy) = ix == 1 ? F[1, iy] : ix > size(F, 1) ? F[ix-1, iy] : face_x(F, mask, ix, iy)
@inline edge_face_y(F, mask, ix, iy) = iy == 1 ? F[ix, 1] : iy > size(F, 2) ? F[ix, iy-1] : face_y(F, mask, ix, iy)

# The bed carries shear only while the ice presses on it: in an N-dependent law a face with N < 0 (water above
# overburden, routine around moulins in the filled sheet) would get a stress along the sliding direction of size ~|N|,
# which the frictional heat then counted as melt. The laws take max(N, 0), as ISSM's Budd and Schoof laws do.
@inline bed_N(N) = max(N, zero(N))

# taub regularizes on the *joint* sliding speed abs_ub = ‖v_b‖ (not the signed component ub_x/ub_y), then restores
# direction by multiplying T with ub_x/‖v_b‖ (ub_y/‖v_b‖) -- see Kazmierczak et al. 2024 Eq. 1 (Joughin et al.
# 2019's regularized Coulomb law) for the same vectorization. Using the signed component directly broke down at the
# instant a real (not purely +x) velocity field was used: the power-law ratio can go negative, and Julia's `^` throws a
# DomainError for a negative real base with a fractional exponent. ‖v_b‖ >= 0 always, so this can't happen here.
#
# abs_ub is reused rather than re-derived locally from ub_x/ub_y: it's already the correct cell-centered ‖v_b‖
# (compute_abs_ub_kernel!), and valid by construction here: set_initial_conditions!, refresh_state! and every step!
# (run.jl) recompute it from ub_x/ub_y before the head solve, so a coupler that writes new ub_x/ub_y between steps is
# picked up. Code that changes ub_x/ub_y and calls compute_taub_x!/_y!/_xy! outside step! must call compute_abs_ub!
# first, or abs_ub (and therefore taub) is stale.
@inline function taub_x_at(rule, cx, mask, N, ub_x, abs_ub, lambda, ix, iy)
    Nf = bed_N(edge_face_x(N, mask, ix, iy))
    v  = edge_face_x(abs_ub, mask, ix, iy)
    lf = edge_face_x(lambda, mask, ix, iy)
    return basal_shear(rule, Nf, v, lf, map(C -> C[ix, iy], cx), ub_x[ix, iy])
end
@inline function taub_y_at(rule, cy, mask, N, ub_y, abs_ub, lambda, ix, iy)
    Nf = bed_N(edge_face_y(N, mask, ix, iy))
    v  = edge_face_y(abs_ub, mask, ix, iy)
    lf = edge_face_y(lambda, mask, ix, iy)
    return basal_shear(rule, Nf, v, lf, map(C -> C[ix, iy], cy), ub_y[ix, iy])
end

@parallel_indices (ix, iy) function compute_taub_x_kernel!(taub_x, mask, N, ub_x, abs_ub, lambda, rule, cx)
    if ix <= size(taub_x, 1) && iy <= size(taub_x, 2)
        taub_x[ix, iy] = taub_x_at(rule, cx, mask, N, ub_x, abs_ub, lambda, ix, iy)
    end
    return
end

@parallel_indices (ix, iy) function compute_taub_y_kernel!(taub_y, mask, N, ub_y, abs_ub, lambda, rule, cy)
    if ix <= size(taub_y, 1) && iy <= size(taub_y, 2)
        taub_y[ix, iy] = taub_y_at(rule, cy, mask, N, ub_y, abs_ub, lambda, ix, iy)
    end
    return
end

# Fused hot-path version: one launch instead of two (see field_gradients.jl's
# compute_dhdxy! for why passing both differently-shaped face arrays as
# arguments makes ParallelStencil infer the right union launch range).
@parallel_indices (ix, iy) function compute_taub_xy_kernel!(taub_x, taub_y, mask, N, ub_x, ub_y, abs_ub, lambda, rule, cx, cy)
    if ix <= size(taub_x, 1) && iy <= size(taub_x, 2)
        taub_x[ix, iy] = taub_x_at(rule, cx, mask, N, ub_x, abs_ub, lambda, ix, iy)
    end
    if ix <= size(taub_y, 1) && iy <= size(taub_y, 2)
        taub_y[ix, iy] = taub_y_at(rule, cy, mask, N, ub_y, abs_ub, lambda, ix, iy)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Sets up `state.taub_x`/`state.taub_y` for `sl` before the Picard loop starts. A no-op for every
law that recomputes `taub` each Picard iteration from `N`/`ub` (see [`compute_taub_x!`](@ref) etc.);
copies `taub_x`/`taub_y` into `state` once for [`PrescribedSlidingLaw`](@ref) (see its own docstring).
"""
initialize_taub!(::AbstractSlidingLaw, state::State, taub_x::AbstractArray, taub_y::AbstractArray) = state

function initialize_taub!(::PrescribedSlidingLaw, state::State, taub_x::AbstractArray, taub_y::AbstractArray)
    state.taub_x .= taub_x
    state.taub_y .= taub_y
    return state
end

"""
$(TYPEDSIGNATURES)

Updates `s.taub_x` under the sliding law `sl` from the current `s.N`/`s.ub_x`/`s.abs_ub`.
"""
function compute_taub_x!(s::State, p::ModelParameters, sl::AbstractSlidingLaw)
    rule, cx, _ = kernel_args(sl, p)
    @parallel compute_taub_x_kernel!(s.taub_x, s.mask, s.N, s.ub_x, s.abs_ub, s.lambda, rule, cx)
    return s
end

"""
$(TYPEDSIGNATURES)

Updates `s.taub_y` under the sliding law `sl`, the y-face counterpart of `compute_taub_x!`.
"""
function compute_taub_y!(s::State, p::ModelParameters, sl::AbstractSlidingLaw)
    rule, _, cy = kernel_args(sl, p)
    @parallel compute_taub_y_kernel!(s.taub_y, s.mask, s.N, s.ub_y, s.abs_ub, s.lambda, rule, cy)
    return s
end

"""
$(TYPEDSIGNATURES)

Fused version of `compute_taub_x!` + `compute_taub_y!`: one `@parallel` launch instead of two.
"""
function compute_taub_xy!(s::State, p::ModelParameters, sl::AbstractSlidingLaw)
    rule, cx, cy = kernel_args(sl, p)
    @parallel compute_taub_xy_kernel!(s.taub_x, s.taub_y, s.mask, s.N, s.ub_x, s.ub_y, s.abs_ub, s.lambda, rule, cx, cy)
    return s
end

"""
$(TYPEDSIGNATURES)

No-op for [`PrescribedSlidingLaw`](@ref): `taub_x`/`taub_y` were already set once by
[`initialize_taub!`](@ref) and never change.
"""
compute_taub_x!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s
compute_taub_y!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s
compute_taub_xy!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s
