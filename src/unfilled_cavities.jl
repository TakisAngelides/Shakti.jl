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

Marks, via `s.stor = 1/dt` on `GROUNDED` cells, where the gap may be partly empty this step, and records the water present as the step starts (`s.w_old = max(0, b - b_empty) + b_uplift`). A no-op
under [`FilledCavities`](@ref) (`s.stor` stays 0, selecting the legacy equations everywhere).
"""
set_cavity_storage!(s::State, dt, ::FilledCavities) = s
set_cavity_storage!(s::State, dt, ::UnfilledCavities) = (@parallel set_cavity_storage_kernel!(s.stor, s.w_old, s.mask, s.b, s.b_empty, s.b_uplift, inv(dt)); s)

@parallel_indices (ix, iy) function compute_b_w_kernel!(b_w, b, b_empty, b_uplift)
    if ix <= size(b_w, 1) && iy <= size(b_w, 2)
        b_w[ix, iy] = max(b[ix, iy] - b_empty[ix, iy] + b_uplift[ix, iy], zero(eltype(b_w)))
    end
    return
end

"""
$(TYPEDSIGNATURES)

Water depth `s.b_w = b - b_empty + b_uplift` as the step starts (`b_uplift` the lifted water layer under
[`IceUplift`](@ref), zero otherwise), never below 0; exactly `b` while `b_empty` is zero (always, under
[`FilledCavities`](@ref)). The face conductances ([`compute_face_flux!`](@ref)) are built from it, so
an empty cavity neither carries nor draws water. No floor at `min(b, p.b_min)`: a dry cell (`W = 0`)
that kept that conductance drained water it did not have wherever its head stayed above a neighbour's
(a margin cell on a slope next to `LAND`), and [`update_b_empty_budget!`](@ref)'s floor `W >= 0` then
created it: 1.4 m^3/s on Drang Drung at 25 m, 21 times the melt input. Without the floor a dry cell's
outflow vanishes with its water; its row stays regular through the unfilled storage `1/dt`
([`cavity_head_terms`](@ref)), or is a closed cell's once all its faces carry no water. It is the water present at the start of the step,
not at the current head iterate: like `b` in the legacy equations it stays fixed during the Picard
loop. That lag makes the water-depth dependence of the flux explicit in time where the water depth is
free (an unfilled or uplifted cell): a step longer than the time such a cell takes to fill or drain,
about `dx*W/(3|q|)` (minutes at 100 m for a thin free-surface layer), overshoots, and the solution
oscillates from step to step or Picard cycles. Taking the depth at the head iterate in a plain Picard
iteration does not help: the one-cell Picard map then has slope ~ -3(W + dzb)/W, and neither Anderson
acceleration nor [`NewtonJFNKSolver`](@ref) converged on the SHW12 benchmark at dt = 6 h. What does is
the end-of-step depth together with the Newton terms of its dependence on the head,
[`ImplicitUpwind`](@ref) ([`compute_b_w_iterate!`](@ref)). `p` is not used (kept for the call signature).
"""
compute_b_w!(s::State, p::ModelParameters) = (@parallel compute_b_w_kernel!(s.b_w, s.b, s.b_empty, s.b_uplift); s)

# The gap at the end of the step that the head rows assume (linear_solver.jl): the start-of-step gap under
# ExplicitGapCoupling (the original scheme, which lags the gap in the conductance), the gap scheme's update G(N) under
# ConsistentGapCoupling (SchemeGapUpdate), with its slope in the head dG/dh = -rho_w g dG/dN (central difference, as in
# scheme_head_terms) where N follows the head (a filled cell).
# Under the dry floor (dry_floor, run.jl) the gap the conductance takes is the water held at b_min, as compute_b! leaves it:
# the floor conducts whether or not it holds water (its row keeps the water from going below zero).
# DiffusiveGapUpdate (WithDiffusion): its gap update is a global solve, so each cell takes the gap of the last solve,
# State.b_next, and its slope State.db_next (DiffusiveCellGap). The head rows store that gap; the conductance takes the
# smaller of it and the start-of-step gap: lagged where the gap opens (a melt opening at the iterate would close the loop
# W -> K -> q -> mdot -> W inside the Picard iteration, see store_start_of_step_melt!), the row's own water where it closes
# (an unfilled cell then conducts no more water than its row holds).
struct DiffusiveCellGap{F} <: AbstractGapCoupling
    bn::F          # State.b_next
    gbn::F         # State.db_next
    conduct::Bool  # for the conductance (min with the start-of-step gap) or for the head row's water (water_converged)
end
@inline iterate_gap(gc::AbstractGapCoupling, bump, bn, gbn, conduct) = cell_gap(gc, bump)
@inline iterate_gap(::DiffusiveGapUpdate{F}, bump, bn, gbn, conduct) where F = DiffusiveCellGap{F}(bn, gbn, conduct)
@inline end_of_step_gap(gc::DiffusiveCellGap, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) = gc.conduct ? min(b, gc.bn) : gc.bn
@inline end_of_step_gap_slope(gc::DiffusiveCellGap, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav) = gc.conduct && gc.bn >= b ? zero(b) : gc.gbn
@inline end_of_step_gap(::ExplicitGapCoupling, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) = b
@inline end_of_step_gap(gu::SchemeGapUpdate, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) =
    gu.dry_floor ? max(scheme_water(gu, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1), b_min) :
                   scheme_gap(gu, b, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1)
@inline end_of_step_gap_slope(::ExplicitGapCoupling, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav) = zero(b)
@inline function end_of_step_gap_slope(gu::SchemeGapUpdate, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav)
    δ = 1e-6 * max(abs(Nv), oftype(Nv, 1e3))
    Gm = end_of_step_gap(gu, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv - δ, rho_i, n_minus_1)
    Gp = end_of_step_gap(gu, b, b_dry, b_min, b_max, mdot, beta, abs_ub, A, Nv + δ, rho_i, n_minus_1)
    return rho_w * ggrav * max(zero(b), (Gm - Gp) / (2δ))
end

# The end-of-step water depth of a cell at head h and its slope dW/dh, by regime (compute_b_w_iterate!): never below 0.
@inline function iterate_water(gcc, bb, bd, h, z, po, Nv, stor, stor_up, b_min, b_max, rho_w, ggrav, mdot, beta, abs_ub, A, rho_i, n_minus_1)
    x  = h - z                     # pressure head
    xo = po / (rho_w * ggrav)      # pressure head at the overburden
    if stor > 0 && x < 0           # unfilled: the gap at N = po minus the empty depth zb - h
        W = end_of_step_gap(gcc, bb, bd, b_min, b_max, mdot, beta, abs_ub, A, po, rho_i, n_minus_1) + x
        slope = one(bb)
    elseif stor_up > 0 && x > xo   # uplifted: the gap at N = 0 plus the layer h - h_o
        W = end_of_step_gap(gcc, bb, bd, b_min, b_max, mdot, beta, abs_ub, A, zero(bb), rho_i, n_minus_1) + (x - xo)
        slope = one(bb)
    else                           # filled: the gap (the start-of-step one, or G(N) under ConsistentGapCoupling)
        W = end_of_step_gap(gcc, bb, bd, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1)
        slope = end_of_step_gap_slope(gcc, bb, bd, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav)
    end
    return W > 0 ? (W, slope) : (zero(bb), zero(bb)) # no water, no conductance (compute_b_w!)
end

@parallel_indices (ix, iy) function compute_b_w_iterate_kernel!(b_w, dbw_dh, b, h, zb, po, N, stor, stor_up, b_min, b_max, rho_w, ggrav, gc, mdot0, mdot, beta, abs_ub, A_visc, rho_i, n_minus_1, b_dry, bump, b_next, db_next)
    if ix <= size(b_w, 1) && iy <= size(b_w, 2)
        gcc = iterate_gap(gc, bump[ix, iy], b_next[ix, iy], db_next[ix, iy], true) # this cell's bump height (compute_beta!), or its gap solve (DiffusiveGapUpdate)
        args = (b[ix, iy], b_dry[ix, iy], h[ix, iy], zb[ix, iy], po[ix, iy], N[ix, iy], stor[ix, iy], stor_up[ix, iy], b_min, b_max, rho_w, ggrav)
        W0, slope0 = iterate_water(gcc, args..., mdot0[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], rho_i, n_minus_1) # the melt opening of the start of the step
        W1, slope1 = iterate_water(gcc, args..., mdot[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], rho_i, n_minus_1)  # the head row's own water
        b_w[ix, iy]    = W1 < W0 ? W1 : W0 # where the melt has fallen within the step, no more water than the row holds
        dbw_dh[ix, iy] = W1 < W0 ? slope1 : slope0
    end
    return
end

"""
$(TYPEDSIGNATURES)

Water depth at the end of the step as a function of the current head iterate, for [`ImplicitUpwind`](@ref), and its slope
`s.dbw_dh = dW/dh`. With the gap `G` the head rows assume for the end of the step (`gc`): `W = G(p_o) - (zb - h)` where the
gap may be partly empty (`s.stor > 0`, [`UnfilledCavities`](@ref)) and `h < zb`, `W = G(0) + (h - h_o)` where the ice may be
lifted (`s.stor_up > 0`, [`IceUplift`](@ref)) and `h > h_o`, `W = G(N)` otherwise, never below 0 (no floor, see [`compute_b_w!`](@ref)). These are
the head rows' own water: the depth they conserve (with the melt opening of `G` from the start of the step,
[`store_start_of_step_melt!`](@ref), or at the iterate where that gives less water). Under [`ExplicitGapCoupling`](@ref) `G` is the start-of-step gap `b`
(the head rows add the gap change only to the water budget), so a filled cell's depth is `b` with slope 0 (the gap is lagged
in its conductance, as in the original scheme) and the unfilled and uplifted depths move one to one with the head. Under
[`ConsistentGapCoupling`](@ref) `G` is the gap scheme's own update: the conductance of every cell then uses the end-of-step
water depth, a filled cell's with slope `rho_w g |dG/dN|`, and the coupled head-gap step is implicit (no lag of the gap in
the conductance either). Both are continuous across the regimes and equal `b` exactly under [`FilledCavities`](@ref),
[`NoUplift`](@ref) and [`ExplicitGapCoupling`](@ref).
"""
compute_b_w_iterate!(s::State, p::ModelParameters, gc::AbstractGapCoupling = ExplicitGapCoupling()) =
    (@parallel compute_b_w_iterate_kernel!(s.b_w, s.dbw_dh, s.b, s.h, s.zb, s.po, s.N, s.stor, s.stor_up, p.b_min, p.b_max, p.rho_w, p.g, gc, s.mdot0, s.mdot, s.beta, s.abs_ub, s.A_visc, p.rho_i, p.n_minus_1_exp, s.b_dry, s.bump, s.b_next, s.db_next); s)

"""
$(TYPEDSIGNATURES)

Keeps the melt rate at the start of the step (`s.mdot0`), for the end-of-step gap in the conductance of
[`ImplicitUpwind`](@ref) ([`compute_b_w_iterate!`](@ref)); a no-op for every other K-face scheme. That gap takes its melt
opening `dt*mdot/rho_i` from the start of the step and only its creep closure from the head iterate: the dissipation melt
grows with the flux, so a melt opening at the iterate would close the loop `W -> K -> q -> mdot -> W` inside the Picard
iteration, a positive feedback (the growth of a channel) with a gain of order `dt*(3/W)*mdot/rho_i`, which the iteration
cannot follow at large `dt` (seen on a 3 m^3/s moulin at dt = 3 h). The head rows keep the melt at the iterate, so the water
balance and its consistency with the gap update are unaffected; only the conductance lags the melt by one step, as the
original scheme lags the whole gap. Where the melt falls within the step the conductance takes the head row's own water,
which is then the smaller ([`compute_b_w_iterate!`](@ref)): with the start-of-step melt a nearly empty cell kept a
conductance for water its row had already drained below zero, and the floor of [`update_b_empty_budget!`](@ref) made up
the difference (Helheim at 500 m, dt = 1 day, cold start: the dissipation melt halved within the step, a gap 1.7 mm
larger, 9e-4 of the melt created, and the step never passed [`water_converged`](@ref)).
"""
store_start_of_step_melt!(::AbstractKFaceScheme, s::State) = s
store_start_of_step_melt!(::ImplicitUpwind, s::State) = (copyto!(s.mdot0, s.mdot); s)

# An outflow-only face (compute_face_masks!) is a kink of the head row like the bed: its flux is K (h - hb)/dx while
# h >= hb and 0 below. Stop an iterate that crosses the boundary head hb there, the face closed on the way down and open
# on the way up, so that each linear solve sees the face in the state it ends in.
@inline function stop_at_boundary_head(hn, hp, hb, ins)
    hp >= hb && hn < hb && return hb - ins # open -> closed: stop just below the boundary head
    hp < hb && hn > hb && return hb        # closed -> open: stop at it
    return hn
end

@parallel_indices (ix, iy) function limit_regime_crossing_kernel!(h, h_prev, dh_it, b, zb, po, stor, stor_up, rho_w, ggrav, filled_cap, mask, land, ocean)
    if ix <= size(h, 1) && iy <= size(h, 2)
        hn, hp, z = h[ix, iy], h_prev[ix, iy], zb[ix, iy]
        ins = max(1e-6 * b[ix, iy], 64 * eps(abs(z) + one(z))) # a step inside the new regime, keeping W ~ b
        bc = b[ix, iy]
        rg = rho_w * ggrav
        if filled_cap && (stor[ix, iy] > 0 || stor_up[ix, iy] > 0) && !(stor[ix, iy] > 0 && hp < z) && !(stor_up[ix, iy] > 0 && hp > z + po[ix, iy] / rg)
            cap = max(max(abs(po[ix, iy] - rg * (hp - z)) / 2, po[ix, iy] / 4) / rg, one(hn)) # from a filled cell: |dN| <= max(|N|/2, po/4), as limit_filled_step!
            hn = clamp(hn, hp - cap, hp + cap)
        end
        if stor[ix, iy] > 0
            hp >= z && hn < z && (hn = z - ins) # filled -> unfilled: stop just below the bed (the unfilled row next)
            hp < z && hn > z && (hn = z)        # unfilled -> filled: stop at the bed (the filled row next)
            hp < z && (hn = max(hn, hp - max(3 * bc / 4, z - hp))) # within unfilled: the head's depth below the bed at most doubles per iteration, or by 3/4 of the gap
        end
        if stor_up[ix, iy] > 0
            ho = z + po[ix, iy] / (rho_w * ggrav)
            hp <= ho && hn > ho && (hn = ho + ins) # filled -> uplifted
            hp > ho && hn < ho && (hn = ho)        # uplifted -> filled
            hp > ho && (hn = min(hn, hp + (bc + hp - ho))) # within uplifted: the water depth at most doubles per iteration
        end
        if (land || ocean) && mask[ix, iy] == GROUNDED # outflow-only LAND/OCEAN faces (compute_face_masks!): open while h >= the boundary head
            ix > 1          && closes_inflow(mask[ix-1, iy], land, ocean) && (hn = stop_at_boundary_head(hn, hp, h[ix-1, iy], ins))
            ix < size(h, 1) && closes_inflow(mask[ix+1, iy], land, ocean) && (hn = stop_at_boundary_head(hn, hp, h[ix+1, iy], ins))
            iy > 1          && closes_inflow(mask[ix, iy-1], land, ocean) && (hn = stop_at_boundary_head(hn, hp, h[ix, iy-1], ins))
            iy < size(h, 2) && closes_inflow(mask[ix, iy+1], land, ocean) && (hn = stop_at_boundary_head(hn, hp, h[ix, iy+1], ins))
        end
        d, dl = hn - hp, dh_it[ix, iy]
        d * dl < 0 && abs(d) > abs(dl) / 2 && (hn = hp + copysign(abs(dl) / 2, d)) # step reverses without contracting: at most half the previous one
        dh_it[ix, iy] = hn - hp
        h[ix, iy] = hn
    end
    return
end

"""
$(TYPEDSIGNATURES)

Stops a head iterate at the regime boundary it crosses (the bed `zb` between the unfilled and filled rows, the overburden
head `h_o` between the filled and uplifted rows), just inside the new regime, under [`ImplicitUpwind`](@ref); a no-op
for every other K-face scheme. Called after each linear solve of the head ([`Picard_iteration!`](@ref),
[`Parabolic_iteration!`](@ref)). The head row's storage slope jumps at these boundaries, from `1/dt` (an unfilled or
uplifted cell, plus the Newton terms of its water depth) to the creep derivative of a filled cell (orders of magnitude
smaller, and zero at `N = 0`), so one Newton step from the filled side lands far into the other regime, where the water
depth hits its floor and the Newton terms vanish, and the next step lands far back: the iteration cycles (seen on SHW12
at dt = 1 h). From a point at the boundary the row is convex on the unfilled and uplifted side and concave on the filled
side, so Newton then approaches the root without crossing back. The same remedy as the "chopping" of phase transitions in
fully implicit reservoir simulators. Inside the unfilled and uplifted regimes the step is limited too, as reservoir
simulators limit the saturation change per Newton iteration: an uplifted cell's water depth at most doubles per
iteration, and the head of an unfilled cell sinks below the bed at most twice as deep per iteration, or by 3/4 of the gap if
that is more (the gap empties in two iterations; below `zb - b` the water depth is at its floor and the head of a dry cell
reaches its value, which can be tens of metres below the bed, in about fifteen iterations instead of one step of 3/4 of the
gap at a time). The factor 2 costs no time against 4 on Drang Drung (dt = 1 h to 1 day) and gives the same solution; at 8
a dt = 1 day run fails (a singular head matrix after a runaway step). Without it, a
Newton step from a thin layer, where the conductance and its Newton terms are small, can land kilometres away (seen on
Drang Drung at dt = 1 h: one cell 1.1 km below the bed, then a neighbour 67 km above the overburden, then overflow). A
converged head is a fixed point of the limited iteration too, so the solution is unchanged. Last, a step that reverses
the previous iteration's step of the same cell is at most half of it (the "oscillation damping" of reservoir simulators):
near a root Newton's steps shrink much faster than that and the damping never binds; it binds only where the iteration
cycles, at a kink of the row (the bed or the overburden head, where the slope of the water depth jumps, or a face whose
upwind donor flips), and there it brackets the root and halves the bracket each iteration, as bisection does (seen on
Drang Drung at dt = 24 h: a cell just below the bed whose Newton step from W = b empties three quarters of the gap, and
from there refills it, every iteration). The previous steps (`s.dh_it`) are zeroed at the start of each step.

Under [`ConsistentGapCoupling`](@ref) (`gc` a [`SchemeGapUpdate`](@ref)) the step of a filled cell is also capped at
`|dN| <= max(|N|/2, po/4)`, the cap of [`limit_filled_step!`](@ref). On thick ice the gap update closes a thin gap within
one step for most `N`, so at `N` near `po` it is nearly flat in `N` and the Newton step of a filled row overshoots by
kilometres; the iteration then jumped between the bed and the overburden head every iteration (synthetic slab under 1.5 to
3 km of ice, dt = 1 h: 500 of 500 iterations from step 5, and the water budget off by 10 to 100 times the melt). With the
cap it converges in 3 to 16 iterations at dt = 1 h and 1 day and closes the budget. [`ExplicitGapCoupling`](@ref), whose
creep term is steep in `N`, is left unchanged.
"""
limit_regime_crossing!(::AbstractKFaceScheme, s::State, h_prev, p::ModelParameters, gc::AbstractGapCoupling = ExplicitGapCoupling()) = s
limit_regime_crossing!(::ImplicitUpwind, s::State, h_prev, p::ModelParameters, gc::AbstractGapCoupling = ExplicitGapCoupling()) =
    (@parallel limit_regime_crossing_kernel!(s.h, h_prev, s.dh_it, s.b, s.zb, s.po, s.stor, s.stor_up, p.rho_w, p.g, gc isa Union{SchemeGapUpdate, DiffusiveGapUpdate}, s.mask, p.outflow_only_land, p.outflow_only_ocean); s)

@parallel_indices (ix, iy) function limit_filled_step_kernel!(h, h_prev, dh_it, mask, zb, po, rho_w, ggrav)
    if ix <= size(h, 1) && iy <= size(h, 2) && mask[ix, iy] == GROUNDED
        hn, hp = h[ix, iy], h_prev[ix, iy]
        rg = rho_w * ggrav
        Np = po[ix, iy] - rg * (hp - zb[ix, iy])                         # N at the previous iterate
        cap = max(max(abs(Np) / 2, po[ix, iy] / 4) / rg, one(hn))        # |dN| <= max(|N|/2, po/4), and at least 1 m of head
        d = clamp(hn - hp, -cap, cap)
        dmax = max((Np - (min(Np, zero(Np)) - po[ix, iy] / 4)) / rg, one(hn)) # N below zero: at most po/4 past min(N_prev, 0) per iteration (linear, not geometric)
        d = min(d, dmax)
        dl = dh_it[ix, iy]
        d * dl < 0 && abs(d) > abs(dl) / 2 && (d = copysign(abs(dl) / 2, d)) # step reverses without contracting: at most half the previous one
        dh_it[ix, iy] = d
        h[ix, iy] = hp + d
    end
    return
end

"""
$(TYPEDSIGNATURES)

Limits each `GROUNDED` cell's head step per Picard iteration under the dry floor ([`dry_floor`](@ref), `gc.dry_floor`)
with a K-face scheme other than [`ImplicitUpwind`](@ref) (which has [`limit_regime_crossing!`](@ref)); a no-op otherwise.
The effective pressure may change by at most half of itself, or a quarter of the overburden near `N = 0`, and a step that
reverses the previous one without contracting is halved (the oscillation damping of [`limit_regime_crossing!`](@ref)).
On the negative side (`N < 0`, head above the overburden, which no physical solution of a filled thick-ice cell reaches) the
relative cap would let `N` run away geometrically (`-5e6`, `-6e7`, `-6.5e8`, ...): `N` may then fall at most `po/4` below
`min(N_prev, 0)` per iteration. Without that, the first raw iterate of the stuck-cells fixture (`N` around `-1e16`) made the
exponential branch of [`relax_update`](@ref) overflow (NaN without this limiter) and, limited, stalled at `N` near `-1e9`.
Without it a cold start on thick ice diverged at the first step (Greenland and Antarctica 16 km, Thwaites 2 km at
`dt = 1 h`): where creep closes the gap within a step the gap update is nearly flat in `N`, the first linear solve
pushes the melt through the floor's conductance, and the head overshoots by kilometres, past the overburden; there the gap
sits at `b_max`, opening no longer takes up the melt, and the dissipation melt feeds itself to NaN. With the limit every
step converged on all three at `dt = 1 h` (30 days), `6 h` and `24 h` (90 days), in at most about 30 iterations. Relative
to `N`, the limit moves the high-suction cells (`N` up to thousands of times `po`) geometrically: a cap of half the
overburden head instead left the first step of Antarctica and Thwaites unconverged after 500 iterations and failed on
Thwaites at `dt = 24 h`. Tested: `|N|/4` to `3|N|/4` and `po/10` to `po/4` behave alike; `po/2` near `N = 0` fails on
Thwaites at `dt = 24 h`, and without the damping Greenland does not converge. A converged head is a fixed point of the
limited iteration, so the solution is unchanged (within the Picard tolerance); warm starts cost about the same. The
extrapolated head guess of a step is not limited: that doubled the iterations on Drang Drung (25 m) and Helheim (200 m).
"""
limit_filled_step!(::AbstractKFaceScheme, ::AbstractGapCoupling, s::State, h_prev, p::ModelParameters) = s
limit_filled_step!(::ImplicitUpwind, ::SchemeGapUpdate, s::State, h_prev, p::ModelParameters) = s
limit_filled_step!(::AbstractKFaceScheme, gu::SchemeGapUpdate, s::State, h_prev, p::ModelParameters) =
    gu.dry_floor ? (@parallel limit_filled_step_kernel!(s.h, h_prev, s.dh_it, s.mask, s.zb, s.po, p.rho_w, p.g); s) : s

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

# Per GROUNDED cell: the head row's end-of-step water minus the water the fluxes and sources leave, w_old + dt (mdot/rho_w
# + ieb - div q), as update_b_empty_budget! takes it (water_converged, elliptic_solver.jl). The row's water is that of
# compute_b_w_iterate! but with the current melt rate: s.b_w takes the gap's melt opening from the start of the step (mdot0).
# With `relative`, divided by the sum of the magnitudes of the cell's terms, the scale of its round-off.
@parallel_indices (ix, iy) function water_budget_residual_kernel!(r, relative, mask, w_old, q_x, q_y, mdot, ieb, dx, dy, dt, rho_w, b, h, zb, po, N, stor, stor_up, b_min, b_max, ggrav, gc, beta, abs_ub, A_visc, rho_i, n_minus_1, b_dry, bump, b_next)
    if ix <= size(r, 1) && iy <= size(r, 2)
        if mask[ix, iy] == GROUNDED
            div = (q_x[ix+1, iy] - q_x[ix, iy]) / dx + (q_y[ix, iy+1] - q_y[ix, iy]) / dy
            W, _ = iterate_water(iterate_gap(gc, bump[ix, iy], b_next[ix, iy], zero(dt), false), b[ix, iy], b_dry[ix, iy], h[ix, iy], zb[ix, iy], po[ix, iy], N[ix, iy], stor[ix, iy], stor_up[ix, iy],
                                 b_min, b_max, rho_w, ggrav, mdot[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], rho_i, n_minus_1)
            res = W - (w_old[ix, iy] + dt * (mdot[ix, iy] / rho_w + ieb[ix, iy] - div))
            scale = W + abs(w_old[ix, iy]) + dt * (abs(mdot[ix, iy] / rho_w) + abs(ieb[ix, iy]) + (abs(q_x[ix+1, iy]) + abs(q_x[ix, iy])) / dx + (abs(q_y[ix, iy+1]) + abs(q_y[ix, iy])) / dy)
            r[ix, iy] = relative ? res / (scale + floatmin(scale)) : res
        else
            r[ix, iy] = zero(dt)
        end
    end
    return
end

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
