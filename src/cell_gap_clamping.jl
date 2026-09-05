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
struct CellGapClamping{F <: AbstractFloat} <: AbstractCellGapClamping
    bounds::Dict{Tuple{Int, Int}, Tuple{F, F}}
end

"""
$(TYPEDSIGNATURES)

Applies `cgc`'s per-cell overrides to `s.b`, called once per timestep after [`compute_b!`](@ref) --
a no-op under [`NoCellGapClamping`](@ref).
"""
apply_cell_gap_clamping!(s::State, ::NoCellGapClamping) = s

# A host round-trip (Array(s.b) ... Data.Array(...)) rather than scalar getindex!/setindex! directly
# on s.b: GPUArrays.jl disallows element-by-element indexing on GPU-resident arrays by default (same
# constraint noted in initial_conditions.jl), and `bounds` is expected to be a short, user-curated
# list (a handful of known-problem cells, not a per-cell field), so the round-trip's cost is
# negligible against a whole timestep -- there's no need for a GPU kernel over what's really a sparse,
# occasional override.
function apply_cell_gap_clamping!(s::State, cgc::CellGapClamping)
    b = Array(s.b)
    for ((i, j), (bmin, bmax)) in cgc.bounds
        b[i, j] = clamp(b[i, j], bmin, bmax)
    end
    s.b .= Data.Array(b)
    return s
end
