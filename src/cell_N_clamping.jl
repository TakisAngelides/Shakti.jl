"""
$(TYPEDSIGNATURES)

Optional per-cell override of `ModelParameters`' global `N_min`/`N_max` effective-pressure clamp --
multiple dispatch on the concrete subtype picks [`NoCellNClamping`](@ref) (no-op, default),
[`CellNClamping`](@ref) (clamps specific `(i, j)` cells to their own `(nmin, nmax)` via a hard
`clamp`), applied via [`apply_cell_N_clamping!`](@ref) every time
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

# The (lowest floor, highest cap) a clamp imposes anywhere, read from the host-side bounds: (-Inf, Inf) for none.
N_clamp_range(::NoCellNClamping) = (-Inf, Inf)
N_clamp_range(c::CellNClamping) = isempty(c.bounds) ? (-Inf, Inf) : (maximum(first, values(c.bounds)), minimum(last, values(c.bounds)))

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
