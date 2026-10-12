# =============================================================================
# Unfilled cavities (free-surface regime)
# =============================================================================
# Where the gap opens faster than water can fill it, the extra volume is empty space holding no
# water and the water pressure is atmospheric, p_w = 0 (Schoof, Hewitt & Werder 2012, JFM 702,
# Part 1, Sec. 4.1, region Omega^e: water depth h_w < gap height h; Wells et al. 2026, GlaDS-2,
# Eq. 1-4 for a single closure covering all regimes). Shakti's legacy equations assume the gap is
# always full of water, which forces N above the overburden (p_w < 0) wherever sliding opens the
# gap faster than inflow, conductance and creep closure can follow.
#
# `State.b_empty` holds the empty part of the gap per unit bed area [m] (>= 0), so the water
# present is `b - b_empty`. Under FilledCavities (default) nothing changes and `b_empty` stays 0.

"""
$(TYPEDSIGNATURES)

Whether the gap may be partly empty of water -- multiple dispatch on the concrete subtype
([`FilledCavities`](@ref)/[`UnfilledCavities`](@ref)), chosen once by the caller of `Simulation`
(keyword `cavity_filling`) -- same "dispatch on a type decided once outside the hot loop" idiom as
[`MeltTerms`](@ref) (`melt_rate.jl`) and [`AbstractOpenBySlidingScheme`](@ref) (`gap_height.jl`).
"""
abstract type AbstractCavityFilling end

"""
$(TYPEDSIGNATURES)

Legacy behavior: the gap is always completely full of water. `State.b_empty` stays zero.
"""
struct FilledCavities <: AbstractCavityFilling end

"""
$(TYPEDSIGNATURES)

The gap may be partly empty (free-surface regime, `p_w = 0`), see this file's header. A `GROUNDED`
cell whose head falls below the bed takes the unfilled head row ([`cavity_head_terms`](@ref),
`linear_solver.jl`): `N = po`, and the empty depth `zb - h` is the cell's storage. `State.b_empty`
records the empty depth (from the water budget after the gap update, [`update_b_empty_budget!`](@ref)),
the face conductances use the water depth `b - b_empty` ([`compute_b_w!`](@ref)), and `p_w` is
floored at 0.
"""
struct UnfilledCavities <: AbstractCavityFilling end

@parallel_indices (ix, iy) function update_b_empty_kernel!(b_empty, mask, h, zb)
    if ix <= size(b_empty, 1) && iy <= size(b_empty, 2)
        b_empty[ix, iy] = (mask[ix, iy] == GROUNDED) * max(zero(eltype(b_empty)), zb[ix, iy] - h[ix, iy])
    end
    return
end

"""
$(TYPEDSIGNATURES)

Sets `s.b_empty` from the solved head: `max(0, zb - h)` on `GROUNDED` cells, `0` elsewhere. A head
below the bed (`p_w < 0` in the legacy equations) is read as the empty depth of the gap, in metres
of water. A no-op under [`FilledCavities`](@ref).
"""
update_b_empty!(s::State, ::FilledCavities) = s
update_b_empty!(s::State, ::UnfilledCavities) = (@parallel update_b_empty_kernel!(s.b_empty, s.mask, s.h, s.zb); s)

@parallel_indices (ix, iy) function set_cavity_storage_kernel!(stor, w_old, mask, b, b_empty, b_uplift, inv_dt)
    if ix <= size(stor, 1) && iy <= size(stor, 2)
        stor[ix, iy] = (mask[ix, iy] == GROUNDED) * inv_dt
        w_old[ix, iy] = max(zero(inv_dt), b[ix, iy] - b_empty[ix, iy]) + b_uplift[ix, iy] # the water present as the step starts (b_uplift: IceUplift, zero otherwise)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Marks, via `s.stor = 1/dt` on `GROUNDED` cells, where the gap may be partly empty this step, and records the water present as the step starts (`s.w_old = b - b_empty`). A no-op
under [`FilledCavities`](@ref) (`s.stor` stays 0, selecting the legacy equations everywhere).
"""
set_cavity_storage!(s::State, dt, ::FilledCavities) = s
set_cavity_storage!(s::State, dt, ::UnfilledCavities) = (@parallel set_cavity_storage_kernel!(s.stor, s.w_old, s.mask, s.b, s.b_empty, s.b_uplift, inv(dt)); s)

@parallel_indices (ix, iy) function compute_b_w_kernel!(b_w, b, b_empty, b_min, b_uplift)
    if ix <= size(b_w, 1) && iy <= size(b_w, 2)
        bb = b[ix, iy]
        b_w[ix, iy] = max(bb - b_empty[ix, iy] + b_uplift[ix, iy], min(bb, b_min))
    end
    return
end

"""
$(TYPEDSIGNATURES)

Water depth `s.b_w = b - b_empty + b_uplift` as the step starts (`b_uplift` the lifted water layer under
[`IceUplift`](@ref), zero otherwise), never below `min(b, p.b_min)` so an empty
cavity still conducts a little; exactly `b` while `b_empty` is zero (always, under
[`FilledCavities`](@ref)). The face conductances ([`compute_face_flux!`](@ref)) are built from it, so
an empty cavity neither carries nor draws water. It is deliberately the water present at the start
of the step, not at the current head iterate: like `b` in the legacy equations it stays fixed during
the Picard loop, since a cubic conductance that follows the unknown makes the iteration oscillate.
That lag makes the water-depth dependence of the flux explicit in time where the water depth is free
(an unfilled or uplifted cell): a step longer than the time such a cell takes to fill or drain, about
`dx*W/(3|q|)` (minutes at 100 m for a thin free-surface layer), overshoots, and the solution oscillates
from step to step or Picard cycles. Taking the depth at the head iterate instead does not help: the
one-cell Picard map then has slope ~ -3(W + dzb)/W, and neither Anderson acceleration nor
[`NewtonJFNKSolver`](@ref) converged on the SHW12 benchmark at dt = 6 h.
"""
compute_b_w!(s::State, p::ModelParameters) = (@parallel compute_b_w_kernel!(s.b_w, s.b, s.b_empty, p.b_min, s.b_uplift); s)

@parallel_indices (ix, iy) function clamp_b_empty_kernel!(b_empty, b)
    if ix <= size(b_empty, 1) && iy <= size(b_empty, 2)
        b_empty[ix, iy] = min(b_empty[ix, iy], b[ix, iy])
    end
    return
end

"""
$(TYPEDSIGNATURES)

Caps `s.b_empty` at the gap height `s.b` (the water present, `b - b_empty`, cannot be negative); called
after the gap update. A safety only: it acts when the head solve left a cell more than empty, e.g.
after a Picard loop that did not converge. A no-op under [`FilledCavities`](@ref).
"""
clamp_b_empty!(s::State, ::FilledCavities) = s
clamp_b_empty!(s::State, ::UnfilledCavities) = (@parallel clamp_b_empty_kernel!(s.b_empty, s.b); s)

@parallel_indices (ix, iy) function update_b_empty_budget_kernel!(b_empty, mask, b, w_old, q_x, q_y, mdot, ieb, dx, dy, dt, rho_w, b_uplift, stor_up, h, zb, po, ggrav)
    if ix <= size(b_empty, 1) && iy <= size(b_empty, 2)
        if mask[ix, iy] == GROUNDED
            div = (q_x[ix+1, iy] - q_x[ix, iy]) / dx + (q_y[ix, iy+1] - q_y[ix, iy]) / dy
            W = max(zero(dt), w_old[ix, iy] + dt * (mdot[ix, iy] / rho_w + ieb[ix, iy] - div)) # water now, from the water budget
            bb = b[ix, iy]
            if stor_up[ix, iy] > 0 && h[ix, iy] > zb[ix, iy] + po[ix, iy] / (rho_w * ggrav)
                # uplifted cell (ice_uplift.jl): the water beyond the gap is the uplift layer
                b_uplift[ix, iy] = max(zero(dt), W - bb)
                b_empty[ix, iy] = clamp(bb - W, zero(dt), bb)
            else
                b_empty[ix, iy] = clamp(bb - W, -bb, bb)
            end
        else
            b_empty[ix, iy] = zero(dt)
        end
    end
    return
end

"""
$(TYPEDSIGNATURES)

Sets `s.b_empty` from the water budget instead of from the head: the water now is `W = w_old + dt*(mdot/rho_w
+ ieb - div(q))` (what the solve's fluxes and sources actually delivered, so the budget closes for every
cell and every gap scheme), and `b_empty = b - W` with the gap `b` as just updated. Where the gap update
and the head equation disagree (the implicit gap schemes change the gap by a different amount than the
head equation assumed, a held `b_max`/`b_min`, ...) the difference is kept in `b_empty` instead of being
lost: positive when the gap is larger than the water (partly empty), negative when there is more water
than gap (stored above the gap volume, released as a source the next step). `W` is floored at 0, and
`b_empty` limited to `[-b, b]`; water beyond these limits is the only water the bookkeeping can still
create or lose. Under [`IceUplift`](@ref), in a cell whose head is above the overburden head the water
beyond the gap is the uplift layer instead, `b_uplift = W - b` (no upper limit), and `b_empty >= 0`. Called after the gap update; the head-based [`update_b_empty!`](@ref) stays for the
parabolic scheme, whose storage terms the water budget above does not include. A no-op under
[`FilledCavities`](@ref).
"""
update_b_empty_budget!(s::State, g::Grid, p::ModelParameters, dt, ::FilledCavities) = s
update_b_empty_budget!(s::State, g::Grid, p::ModelParameters, dt, ::UnfilledCavities) =
    (@parallel update_b_empty_budget_kernel!(s.b_empty, s.mask, s.b, s.w_old, s.q_x, s.q_y, s.mdot, s.ieb, g.dx, g.dy, dt, p.rho_w, s.b_uplift, s.stor_up, s.h, s.zb, s.po, p.g); s)
