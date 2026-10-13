# =============================================================================
# Ice uplift (water pressure capped at the overburden)
# =============================================================================
# Without it Shakti lets the water pressure exceed the ice overburden (N < 0, N_min = -Inf by default):
# creep then turns into opening at a rate proportional to the gap, which no steady state balances, and a
# coupled ice model receives negative effective pressure. In reality the ice is lifted off the bed
# (hydraulic jacking) once p_w reaches p_o, and extra water is stored in the lifted space at p_w = p_o
# (Schoof, Hewitt & Werder 2012, Part 1, region Omega^f; Wells et al. 2026, GlaDS-2, Eq. 4 upper branch,
# with the ice treated as hydrostatic and static, so p_o does not change with the uplift).
#
# This is the mirror of the unfilled cavities (unfilled_cavities.jl) at the other bound. With the
# overburden head h_o = zb + p_o/(rho_w g), a GROUNDED cell whose head rises above h_o has
#     p_w = p_o, N = 0 (no creep closure; the gap opens only by sliding and melt),
#     uplift layer b_uplift = h - h_o [m of water], stored with coefficient 1/dt,
# so its water balance (W_new - W_old)/dt + div(q) = mdot/rho_w + ieb, with W = b - b_empty + b_uplift the
# water per bed area, gives the head row
#     div(q) + h/dt = mdot/rho_w + ieb - rate_up - b_empty_old/dt + b_uplift_old/dt + h_o/dt,
#     rate_up = mdot/rho_i + beta*|u_b| (db/dt at N = 0).
# At h = h_o it equals the filled row (which then has N = 0 too), so the cell lifts off continuously.
# The filled and unfilled rows get the source + b_uplift_old/dt: water lifted last step is released first.
# The row is chosen from the previous Picard iterate (h > h_o), as for the unfilled row; its diagonal 1/dt
# replaces the creep derivative, which vanishes at N = 0. The flux through an uplifted cell uses the water
# layer b + b_uplift for its conductance, and its head h = h_o + b_uplift already contains the thickness of
# the lifted water (with SheetTopPotential the potential is h + b, GlaDS-2 Eq. 7 exactly).
#
# State.stor_up = 1/dt marks cells where uplift is allowed (GROUNDED under IceUplift, 0 otherwise: the
# legacy rows, bit for bit); State.b_uplift is the uplift layer, set from the head after the solve
# (update_b_uplift!) and, with unfilled cavities on the elliptic scheme, from the water budget
# (update_b_empty_budget!, unfilled_cavities.jl).

"""
$(TYPEDSIGNATURES)

Whether the ice may be lifted off the bed when the water pressure reaches the overburden -- multiple
dispatch on the concrete subtype ([`NoUplift`](@ref)/[`IceUplift`](@ref)), chosen once by the caller of
`Simulation` (keyword `uplift`), same idiom as [`AbstractCavityFilling`](@ref).
"""
abstract type AbstractIceUplift end

"""
$(TYPEDSIGNATURES)

Default: no upper bound on the water pressure (`N < 0` allowed, as in the original equations).
"""
struct NoUplift <: AbstractIceUplift end

"""
$(TYPEDSIGNATURES)

The water pressure is capped at the overburden: above it the ice is lifted and the extra water is stored
in an uplift layer at `p_w = p_o`, `N = 0`. See this file's header.
"""
struct IceUplift <: AbstractIceUplift end

@parallel_indices (ix, iy) function set_uplift_storage_kernel!(stor_up, mask, inv_dt)
    if ix <= size(stor_up, 1) && iy <= size(stor_up, 2)
        stor_up[ix, iy] = (mask[ix, iy] == GROUNDED) * inv_dt
    end
    return
end

"""
$(TYPEDSIGNATURES)

Marks, via `s.stor_up = 1/dt` on `GROUNDED` cells, where the ice may be lifted this step. A no-op under
[`NoUplift`](@ref) (`s.stor_up` stays 0, selecting the legacy rows).
"""
set_uplift_storage!(s::State, dt, ::NoUplift) = s
set_uplift_storage!(s::State, dt, ::IceUplift) = (@parallel set_uplift_storage_kernel!(s.stor_up, s.mask, inv(dt)); s)

@parallel_indices (ix, iy) function update_b_uplift_kernel!(b_uplift, mask, h, zb, po, rho_w, ggrav)
    if ix <= size(b_uplift, 1) && iy <= size(b_uplift, 2)
        b_uplift[ix, iy] = (mask[ix, iy] == GROUNDED) * max(zero(eltype(b_uplift)), h[ix, iy] - (zb[ix, iy] + po[ix, iy] / (rho_w * ggrav)))
    end
    return
end

"""
$(TYPEDSIGNATURES)

Sets `s.b_uplift` from the solved head: `max(0, h - h_o)`, `h_o = zb + po/(rho_w g)`, on `GROUNDED` cells.
A no-op under [`NoUplift`](@ref).
"""
update_b_uplift!(s::State, p::ModelParameters, ::NoUplift) = s
update_b_uplift!(s::State, p::ModelParameters, ::IceUplift) = (@parallel update_b_uplift_kernel!(s.b_uplift, s.mask, s.h, s.zb, s.po, p.rho_w, p.g); s)
