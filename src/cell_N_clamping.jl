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
struct CellNClamping{F <: AbstractFloat, I <: AbstractVector, V <: AbstractVector} <: AbstractCellNClamping
    bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}
    idx::I # the cells of `bounds`, as CartesianIndex, on the active backend (built once, see clamp_cells)
    lo::V  # their lower bounds, same order, floattype, on the active backend
    hi::V  # their upper bounds
end

CellNClamping(bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}) where F <: AbstractFloat = CellNClamping(bounds, clamp_cells(bounds)...)

# Backend-resident copy of a small host vector (the per-cell clamp lists): kept on the same device
# as the State fields they index, so clamping is one in-place broadcast through a view -- no
# host round-trip of the whole field, and no scalar indexing of a GPU array.
@static if backend == "CUDA"
    to_backend(x::AbstractVector) = CuArray(x)
elseif backend == "Metal"
    to_backend(x::AbstractVector) = MtlArray(x)
else
    to_backend(x::AbstractVector) = x
end

"""
$(TYPEDSIGNATURES)

Flattens a per-cell `(i, j) => (lo, hi)` bounds dictionary into three backend-resident vectors
(cell indices, lower bounds, upper bounds, in matching order), built once at construction so each
clamp is a single in-place broadcast.
"""
function clamp_cells(bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}) where F <: AbstractFloat
    ks = collect(keys(bounds))
    idx = to_backend([CartesianIndex(k) for k in ks])
    lo = to_backend([floattype(bounds[k][1]) for k in ks])
    hi = to_backend([floattype(bounds[k][2]) for k in ks])
    return idx, lo, hi
end

"""
$(TYPEDSIGNATURES)

Applies `cnc`'s per-cell overrides to `s.N`, called every time [`compute_N!`](@ref) runs -- a no-op
under [`NoCellNClamping`](@ref).
"""
apply_cell_N_clamping!(s::State, ::NoCellNClamping) = s

# In place through a view of just the listed cells (a device-side gather/scatter on GPU): called
# every Picard iteration, so it must not copy the whole field to the host and back.
function apply_cell_N_clamping!(s::State, cnc::CellNClamping)
    Nv = view(s.N, cnc.idx)
    Nv .= clamp.(Nv, cnc.lo, cnc.hi)
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
struct SmoothCellNClamping{F <: AbstractFloat, I <: AbstractVector, V <: AbstractVector, S <: AbstractFloat} <: AbstractCellNClamping
    bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}
    smoothing::F # transition width in N's own units (Pa); larger = smoother/safer for JFNK's Jacobian but a less faithful floor/cap
    idx::I # see CellNClamping
    lo::V
    hi::V
    smoothing_ft::S # `smoothing` in floattype, for the device-side broadcast
end

SmoothCellNClamping(bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}, smoothing) where F <: AbstractFloat =
    SmoothCellNClamping(bounds, F(smoothing), clamp_cells(bounds)..., floattype(smoothing))

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
    Nv = view(s.N, cnc.idx)
    sm = cnc.smoothing_ft
    Nv .= _smooth_cap.(_smooth_floor.(Nv, cnc.lo, sm), cnc.hi, sm)
    return s
end
