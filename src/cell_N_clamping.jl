"""
$(TYPEDSIGNATURES)

Optional per-cell override of `ModelParameters`' global `N_min`/`N_max` effective-pressure clamp --
multiple dispatch on the concrete subtype picks [`NoCellNClamping`](@ref) (no-op, default),
[`CellNClamping`](@ref) (clamps specific `(i, j)` cells to their own `(nmin, nmax)` via a hard
`clamp`), or [`SmoothCellNClamping`](@ref) (the same per-cell bounds, but softplus-smoothed --
Newton-only, see its own docstring), applied via [`apply_cell_N_clamping!`](@ref) every time
[`compute_N!`](@ref) runs.

Mirrors [`AbstractCellGapClamping`](@ref)'s design, but N (unlike b) is a derived quantity
recomputed from scratch every Picard iteration (not a persisted state variable), so this has to be
applied inside [`compute_N!`](@ref) itself -- a once-per-timestep post-hoc clamp (the way
[`apply_cell_gap_clamping!`](@ref) follows [`compute_b!`](@ref)) would be inert, since N gets
overwritten from `po - pw` again before the next timestep's first Picard iteration even starts.
Preferred over the global `ModelParameters.N_min`/`N_max` where only specific known-problem cells
need floored/capped `N`, leaving the rest of the domain free to develop real (possibly negative,
possibly channel-driving) `N` excursions rather than flooring them everywhere.
"""
abstract type AbstractCellNClamping end

"""
$(TYPEDSIGNATURES)

No per-cell override: [`apply_cell_N_clamping!`](@ref) is a no-op. [`Simulation`](@ref)'s default.
"""
struct NoCellNClamping <: AbstractCellNClamping end

"""
$(TYPEDSIGNATURES)

Clamps specific grid cells' effective pressure `N` to their own `(nmin, nmax)`, overriding whatever
the global `ModelParameters.N_min`/`N_max` clamp already gave them. `bounds` maps a cell's `(i, j)`
grid index to its `(nmin, nmax)` pair, e.g. `CellNClamping(Dict((209, 38) => (0.0, Inf)))` floors
just that one cell at `N = 0` regardless of the (possibly `-Inf`/`Inf`, i.e. off) global setting.
"""
struct CellNClamping{F <: AbstractFloat} <: AbstractCellNClamping
    bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}
end

"""
$(TYPEDSIGNATURES)

Applies `cnc`'s per-cell overrides to `s.N`, called every time [`compute_N!`](@ref) runs -- a no-op
under [`NoCellNClamping`](@ref).
"""
apply_cell_N_clamping!(s::State, ::NoCellNClamping) = s

# Host round-trip rather than scalar getindex!/setindex! directly on s.N, same reasoning as
# apply_cell_gap_clamping! (cell_gap_clamping.jl): GPUArrays.jl disallows element-by-element
# indexing on GPU-resident arrays by default, and `bounds` is expected to be a short, user-curated
# list of known-problem cells, not a per-cell field -- called every Picard iteration, but still
# negligible against a whole elliptic solve.
function apply_cell_N_clamping!(s::State, cnc::CellNClamping)
    N = Array(s.N)
    for ((i, j), (nmin, nmax)) in cnc.bounds
        N[i, j] = clamp(N[i, j], nmin, nmax)
    end
    s.N .= Data.Array(N)
    return s
end

"""
$(TYPEDSIGNATURES)

Smoothed counterpart to [`CellNClamping`](@ref): floors/caps the same per-cell `(nmin, nmax)`
bounds, but via a softplus-smoothed transition (width `smoothing`, in `N`'s own units, Pa) instead
of a hard `clamp`. As `smoothing -> 0` this converges to [`CellNClamping`](@ref)'s exact hard clamp
-- the whole point of keeping `smoothing` away from 0 is that a hard clamp has a genuine kink (a
discontinuity in `dN/dh`) at each bound it actively floors/caps: once a cell's raw `N` crosses that
boundary, [`CellNClamping`](@ref)'s clamped output stops responding to further changes in `h` at
that cell AT ALL (zero derivative there). This is a confirmed, real cause of Jacobian
ill-conditioning for [`NewtonJFNKSolver`](@ref)'s finite-difference Jacobian-vector products at
exactly these cells (see `newton_solver.jl`'s constructor docstring for the full failure analysis
this type responds to) -- a small residual there does not imply a small solution error, since the
clamped cell's own local sensitivity has vanished. [`PicardSolver`](@ref) never has this problem
(it re-solves the whole linearized system from scratch every iteration rather than inferring
convergence from the residual's local slope), so [`CellNClamping`](@ref)'s existing hard clamp is
intentionally left untouched for Picard-driven solvers (`CholeskyDirectSolver`/`CUDSSDirectSolver`)
-- this type is meant to be constructed and passed ONLY where [`NewtonJFNKSolver`](@ref) is in use,
with the SAME `bounds` a `CellNClamping` would otherwise use for the same dataset.
"""
struct SmoothCellNClamping{F <: AbstractFloat} <: AbstractCellNClamping
    bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}
    smoothing::F # transition width in N's own units (Pa); larger = smoother/safer for JFNK's Jacobian but a less faithful floor/cap
end

# Numerically stable softplus, log(1+exp(z)), avoiding overflow at large z via the identity
# softplus(z) = z + softplus(-z) (only ever evaluates exp() at a non-positive argument).
_stable_softplus(z::F) where F <: AbstractFloat = z > zero(F) ? z + log1p(exp(-z)) : log1p(exp(z))

# Smoothly floors x toward nmin; a no-op when nmin == -Inf, matching clamp(x, -Inf, nmax)'s own
# no-op floor. As smoothing -> 0, converges to max(x, nmin).
function _smooth_floor(x::F, nmin::F, smoothing::F) where F <: AbstractFloat
    isinf(nmin) && return x
    return nmin + _stable_softplus((x - nmin) / smoothing) * smoothing
end

# Smoothly caps x toward nmax; a no-op when nmax == Inf. As smoothing -> 0, converges to min(x, nmax).
function _smooth_cap(x::F, nmax::F, smoothing::F) where F <: AbstractFloat
    isinf(nmax) && return x
    return nmax - _stable_softplus((nmax - x) / smoothing) * smoothing
end

"""
$(TYPEDSIGNATURES)

Applies `cnc`'s smoothed per-cell overrides to `s.N`, called every time [`compute_N!`](@ref) runs
-- see [`SmoothCellNClamping`](@ref) for why this exists alongside [`CellNClamping`](@ref)'s own
hard-clamp version of the same operation.
"""
function apply_cell_N_clamping!(s::State, cnc::SmoothCellNClamping)
    N = Array(s.N)
    for ((i, j), (nmin, nmax)) in cnc.bounds
        N[i, j] = _smooth_cap(_smooth_floor(N[i, j], nmin, cnc.smoothing), nmax, cnc.smoothing)
    end
    s.N .= Data.Array(N)
    return s
end
