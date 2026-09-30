"""
$(TYPEDSIGNATURES)

Optional per-cell override of `ModelParameters`' global `b_min`/`b_max` gap-height clamp -- multiple
dispatch on the concrete subtype picks [`NoCellGapClamping`](@ref) (no-op, default) or
[`CellGapClamping`](@ref) (clamps specific `(i, j)` cells to their own `(bmin, bmax)`, applied via
[`apply_cell_gap_clamping!`](@ref) after [`compute_b!`](@ref) each timestep).

Deliberately independent of `gap_height.jl`'s own kernels: the global clamp still runs exactly as
before for every cell, and this only ever adjusts the handful of cells a user explicitly lists --
existing simulations (which all use [`NoCellGapClamping`](@ref), [`Simulation`](@ref)'s default) see
no change in behaviour at all.
"""
abstract type AbstractCellGapClamping end

"""
$(TYPEDSIGNATURES)

No per-cell override: [`apply_cell_gap_clamping!`](@ref) is a no-op. [`Simulation`](@ref)'s default.
"""
struct NoCellGapClamping <: AbstractCellGapClamping end

"""
$(TYPEDSIGNATURES)

Clamps specific grid cells' gap height `b` to their own `(bmin, bmax)`, overriding whatever the
global `ModelParameters.b_min`/`b_max` clamp already gave them that step. `bounds` maps a cell's
`(i, j)` grid index to its `(bmin, bmax)` pair, e.g. `CellGapClamping(Dict((81, 175) => (0.0, 5.0)))`
caps just that one cell at 5m regardless of the global `b_max`.
"""
struct CellGapClamping{F <: AbstractFloat, I <: AbstractVector, V <: AbstractVector} <: AbstractCellGapClamping
    bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}
    idx::I # the cells of `bounds` on the active backend, see clamp_cells (cell_N_clamping.jl)
    lo::V
    hi::V
end

CellGapClamping(bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}) where F <: AbstractFloat = CellGapClamping(bounds, clamp_cells(bounds)...)

"""
$(TYPEDSIGNATURES)

Applies `cgc`'s per-cell overrides to `s.b`, called once per timestep after [`compute_b!`](@ref) --
a no-op under [`NoCellGapClamping`](@ref).
"""
apply_cell_gap_clamping!(s::State, ::NoCellGapClamping) = s

# In place through a view of just the listed cells, same as apply_cell_N_clamping! (cell_N_clamping.jl).
function apply_cell_gap_clamping!(s::State, cgc::CellGapClamping)
    bv = view(s.b, cgc.idx)
    bv .= clamp.(bv, cgc.lo, cgc.hi)
    return s
end
