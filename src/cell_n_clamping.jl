"""
$(TYPEDSIGNATURES)

Optional per-cell override of `ModelParameters`' global `N_min`/`N_max` effective-pressure clamp --
multiple dispatch on the concrete subtype picks [`NoCellNClamping`](@ref) (no-op, default) or
[`CellNClamping`](@ref) (clamps specific `(i, j)` cells to their own `(nmin, nmax)`, applied via
[`apply_cell_n_clamping!`](@ref) every time [`compute_N!`](@ref) runs).

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

No per-cell override: [`apply_cell_n_clamping!`](@ref) is a no-op. [`Simulation`](@ref)'s default.
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
apply_cell_n_clamping!(s::State, ::NoCellNClamping) = s

# Host round-trip rather than scalar getindex!/setindex! directly on s.N, same reasoning as
# apply_cell_gap_clamping! (cell_gap_clamping.jl): GPUArrays.jl disallows element-by-element
# indexing on GPU-resident arrays by default, and `bounds` is expected to be a short, user-curated
# list of known-problem cells, not a per-cell field -- called every Picard iteration, but still
# negligible against a whole elliptic solve.
function apply_cell_n_clamping!(s::State, cnc::CellNClamping)
    N = Array(s.N)
    for ((i, j), (nmin, nmax)) in cnc.bounds
        N[i, j] = clamp(N[i, j], nmin, nmax)
    end
    s.N .= Data.Array(N)
    return s
end
