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
# `State.W` holds the water per unit bed area [m], so the empty part of the gap is `b - W` (`empty_gap`). Under
# FilledCavities (SHAKTI's original sheet) nothing changes and the gap is full, `W = b`.

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

SHAKTI's original sheet: the gap is always completely full of water, `State.W = b` (less the dry floor, `dry_floor`).
"""
struct FilledCavities <: AbstractCavityFilling end

"""
$(TYPEDSIGNATURES)

The gap may be partly empty (free-surface regime, `p_w = 0`), see this file's header. A `GROUNDED`
cell whose head falls below the bed takes the unfilled head row (`cavity_head_terms`,
`linear_solver.jl`): `N = po`, and the empty depth `zb - h` is the cell's storage. The water `State.W` comes
from the water budget after the gap update ([`update_water_budget!`](@ref)), the empty part of the gap is `b - W`,
the face conductances use the water depth `W` ([`compute_b_w!`](@ref)), and `p_w` is
floored at 0. The default of [`Simulation`](@ref) (with [`IceUplift`](@ref): the GlaDS-2 sheet).
"""
struct UnfilledCavities <: AbstractCavityFilling end

@parallel_indices (ix, iy) function set_cavity_storage_kernel!(stor, mask, inv_dt)
    if ix <= size(stor, 1) && iy <= size(stor, 2)
        stor[ix, iy] = (mask[ix, iy] == GROUNDED) * inv_dt
    end
    return
end

"""
$(TYPEDSIGNATURES)

Marks, via `s.stor = 1/dt` on `GROUNDED` cells, where the gap may be partly empty this step. Under
[`FilledCavities`](@ref) it zeroes `s.stor`, selecting the legacy equations everywhere: the kernels read
it whatever the option, so a `State` that ran with unfilled cavities (a spin-up, a checkpoint) would otherwise
keep their rows.
"""
set_cavity_storage!(s::State, dt, ::FilledCavities) = (fill!(s.stor, zero(eltype(s.stor))); s)
set_cavity_storage!(s::State, dt, ::UnfilledCavities) = (@parallel set_cavity_storage_kernel!(s.stor, s.mask, inv(dt)); s)

@parallel_indices (ix, iy) function compute_b_w_kernel!(b_w, W)
    if ix <= size(b_w, 1) && iy <= size(b_w, 2)
        b_w[ix, iy] = max(W[ix, iy], zero(eltype(b_w)))
    end
    return
end

"""
$(TYPEDSIGNATURES)

Water depth `s.b_w = W` as the step starts (`s.W`, the water present: the gap less its empty part under
[`UnfilledCavities`](@ref) or the dry volume the `b_min` clamp keeps under the filled dry floor, plus the lifted layer
under [`IceUplift`](@ref)), never below 0; exactly `b` under SHAKTI's original sheet. The face conductances ([`compute_face_flux!`](@ref)) are built from it, so
an empty cavity neither carries nor draws water. No floor at `min(b, p.b_min)`: a dry cell (`W = 0`)
that kept that conductance drained water it did not have wherever its head stayed above a neighbour's
(a margin cell on a slope next to `LAND`), and [`update_water_budget!`](@ref)'s floor `W >= 0` then
created it: 1.4 m^3/s on Drang Drung at 25 m, 21 times the melt input. Without the floor a dry cell's
outflow vanishes with its water; its row stays regular through the unfilled storage `1/dt`
(`cavity_head_terms`), or is a closed cell's once all its faces carry no water. It is the water present at the start of the step,
not at the current head iterate: like `b` in the legacy equations it stays fixed during the Picard
loop. That lag makes the water-depth dependence of the flux explicit in time where the water depth is
free (an unfilled or uplifted cell): a step longer than the time such a cell takes to fill or drain,
about `dx*W/(3|q|)` (minutes at 100 m for a thin free-surface layer), overshoots, and the solution
oscillates from step to step or Picard cycles. Taking the depth at the head iterate in a plain Picard
iteration does not help: the one-cell Picard map then has slope ~ -3(W + dzb)/W, and neither Anderson
acceleration nor a Jacobian-free Newton-Krylov solver (since removed) converged on the SHW12 benchmark at dt = 6 h. What does is
the end-of-step depth together with the Newton terms of its dependence on the head,
[`ImplicitUpwind`](@ref) ([`compute_b_w_iterate!`](@ref)). `p` is not used (kept for the call signature).
"""
compute_b_w!(s::State, p::ModelParameters) = (@parallel compute_b_w_kernel!(s.b_w, s.W); s)

# The gap at the end of the step that the head rows assume (linear_solver.jl): the start-of-step gap under
# ExplicitGapCoupling (the original scheme, which lags the gap in the conductance), the gap scheme's update G(N) under
# ConsistentGapCoupling (SchemeGapUpdate), with its slope in the head dG/dh = -rho_w g dG/dN (central difference, as in
# scheme_head_terms) where N follows the head (a filled cell).
# Under the dry floor (dry_floor, run.jl) the conductance takes the water the update leaves (scheme_water of W), not the gap held
# at b_min: a drained cell conducting at K(b_min) relayed its neighbours' water at a flat head, lost its storage slope
# (dG/dN -> 0 as W -> 0) and drove p_w to -3 po on Drang Drung (b_min = 1e-3), a state it never left. With K from W the
# results no longer depend on b_min (b_min = 1e-3 matches b_min = 1e-6 to all printed digits there).
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
@inline end_of_step_gap(gc::DiffusiveCellGap, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) = gc.conduct ? min(b, gc.bn) : gc.bn
@inline end_of_step_gap_slope(gc::DiffusiveCellGap, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav) = gc.conduct && gc.bn >= b ? zero(b) : gc.gbn
@inline end_of_step_gap(::ExplicitGapCoupling, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) = b
@inline end_of_step_gap(gu::SchemeGapUpdate, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) =
    gu.dry_floor ? scheme_water(gu, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) :
                   scheme_gap(gu, b, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1)
@inline end_of_step_gap_slope(::ExplicitGapCoupling, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav) = zero(b)
@inline function end_of_step_gap_slope(gu::SchemeGapUpdate, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav)
    δ = oftype(Nv, 1e-6) * max(abs(Nv), oftype(Nv, 1e3))
    Gm = end_of_step_gap(gu, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv - δ, rho_i, n_minus_1)
    Gp = end_of_step_gap(gu, b, W, b_min, b_max, mdot, beta, abs_ub, A, Nv + δ, rho_i, n_minus_1)
    return rho_w * ggrav * max(zero(b), (Gm - Gp) / (2δ))
end

# The end-of-step water depth of a cell at head h and its slope dW/dh, by regime (compute_b_w_iterate!): never below 0.
@inline function iterate_water(gcc, bb, wo, h, z, po, Nv, stor, stor_up, b_min, b_max, rho_w, ggrav, mdot, beta, abs_ub, A, rho_i, n_minus_1)
    x  = h - z                     # pressure head
    xo = po / (rho_w * ggrav)      # pressure head at the overburden
    if stor > 0 && x < 0           # unfilled: the gap at N = po minus the empty depth zb - h
        W = end_of_step_gap(gcc, bb, wo, b_min, b_max, mdot, beta, abs_ub, A, po, rho_i, n_minus_1) + x
        slope = one(bb)
    elseif stor_up > 0 && x > xo   # uplifted: the gap at N = 0 plus the layer h - h_o
        W = end_of_step_gap(gcc, bb, wo, b_min, b_max, mdot, beta, abs_ub, A, zero(bb), rho_i, n_minus_1) + (x - xo)
        slope = one(bb)
    else                           # filled: the gap (the start-of-step one, or G(N) under ConsistentGapCoupling)
        W = end_of_step_gap(gcc, bb, wo, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1)
        slope = end_of_step_gap_slope(gcc, bb, wo, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav)
    end
    return W > 0 ? (W, slope) : (zero(bb), zero(bb)) # no water, no conductance (compute_b_w!)
end

@parallel_indices (ix, iy) function compute_b_w_iterate_kernel!(b_w, dbw_dh, b, h, zb, po, N, stor, stor_up, b_min, b_max, rho_w, ggrav, gc, mdot0, mdot, beta, abs_ub, A_visc, rho_i, n_minus_1, W, bump, b_next, db_next)
    if ix <= size(b_w, 1) && iy <= size(b_w, 2)
        gcc = iterate_gap(gc, bump[ix, iy], b_next[ix, iy], db_next[ix, iy], true) # this cell's bump height (compute_beta!), or its gap solve (DiffusiveGapUpdate)
        args = (b[ix, iy], W[ix, iy], h[ix, iy], zb[ix, iy], po[ix, iy], N[ix, iy], stor[ix, iy], stor_up[ix, iy], b_min, b_max, rho_w, ggrav)
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
`store_start_of_step_melt!`, or at the iterate where that gives less water). Under [`ExplicitGapCoupling`](@ref) `G` is the start-of-step gap `b`
(the head rows add the gap change only to the water budget), so a filled cell's depth is `b` with slope 0 (the gap is lagged
in its conductance, as in the original scheme) and the unfilled and uplifted depths move one to one with the head. Under
[`ConsistentGapCoupling`](@ref) `G` is the gap scheme's own update: the conductance of every cell then uses the end-of-step
water depth, a filled cell's with slope `rho_w g |dG/dN|`, and the coupled head-gap step is implicit (no lag of the gap in
the conductance either). Both are continuous across the regimes and equal `b` exactly under [`FilledCavities`](@ref),
[`NoUplift`](@ref) and [`ExplicitGapCoupling`](@ref).
"""
compute_b_w_iterate!(s::State, p::ModelParameters, gc::AbstractGapCoupling = ExplicitGapCoupling()) =
    (@parallel compute_b_w_iterate_kernel!(s.b_w, s.dbw_dh, s.b, s.h, s.zb, s.po, s.N, s.stor, s.stor_up, p.b_min, p.b_max, p.rho_w, p.g, gc, s.mdot0, s.mdot, s.beta, s.abs_ub, s.A_visc, p.rho_i, p.n_minus_1_exp, s.W, s.bump, s.b_next, s.db_next); s)

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
conductance for water its row had already drained below zero, and the floor of [`update_water_budget!`](@ref) made up
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

@parallel_indices (ix, iy) function limit_head_step_kernel!(h, h_prev, dh_it, b, zb, po, stor, stor_up, rho_w, ggrav, filled_cap, mask, land, ocean, db_x, db_y)
    if ix <= size(h, 1) && iy <= size(h, 2)
        hn, hp, z = h[ix, iy], h_prev[ix, iy], zb[ix, iy]
        ins = max(oftype(z, 1e-6) * b[ix, iy], 64 * eps(abs(z) + one(z))) # a step inside the new regime, keeping W ~ b
        bc = b[ix, iy]
        rg = rho_w * ggrav
        ho = z + po[ix, iy] / rg
        if filled_cap && mask[ix, iy] == GROUNDED && !(stor[ix, iy] > 0 && hp < z) && !(stor_up[ix, iy] > 0 && hp > ho) # a filled cell
            Np = po[ix, iy] - rg * (hp - z)                                       # N at the previous iterate
            cap = max(max(abs(Np) / 2, po[ix, iy] / 4) / rg, one(hn))            # |dN| <= max(|N|/2, po/4), and at least 1 m of head
            hn = clamp(hn, hp - cap, hp + cap)
            stor[ix, iy] == 0 && stor_up[ix, iy] == 0 && (hn = min(hn, hp + max((Np - (min(Np, zero(Np)) - po[ix, iy] / 4)) / rg, one(hn)))) # filled sheet, N below zero: at most po/4 past min(N_prev, 0) per iteration (linear, not geometric)
        end
        if stor[ix, iy] > 0
            hp >= z && hn < z && (hn = z - ins) # filled -> unfilled: stop just below the bed (the unfilled row next)
            hp < z && hn > z && (hn = z)        # unfilled -> filled: stop at the bed (the filled row next)
            hp < z && (hn = max(hn, hp - max(3 * bc / 4, z - hp))) # within unfilled: the head's depth below the bed at most doubles per iteration, or by 3/4 of the gap
        end
        if stor_up[ix, iy] > 0
            hp <= ho && hn > ho && (hn = ho + ins) # filled -> uplifted
            hp > ho && hn < ho && (hn = ho)        # uplifted -> filled
            hp > ho && (hn = min(hn, hp + (bc + hp - ho))) # within uplifted: the water depth at most doubles per iteration
        end
        if (land || ocean) && mask[ix, iy] == GROUNDED # outflow-only LAND/OCEAN faces (compute_face_masks!): open while h + b >= the boundary head (b: the face's gap jump, SheetTopPotential)
            ix > 1          && closes_inflow(mask[ix-1, iy], land, ocean) && (hn = stop_at_boundary_head(hn, hp, h[ix-1, iy] - db_x[ix, iy], ins))
            ix < size(h, 1) && closes_inflow(mask[ix+1, iy], land, ocean) && (hn = stop_at_boundary_head(hn, hp, h[ix+1, iy] + db_x[ix+1, iy], ins))
            iy > 1          && closes_inflow(mask[ix, iy-1], land, ocean) && (hn = stop_at_boundary_head(hn, hp, h[ix, iy-1] - db_y[ix, iy], ins))
            iy < size(h, 2) && closes_inflow(mask[ix, iy+1], land, ocean) && (hn = stop_at_boundary_head(hn, hp, h[ix, iy+1] + db_y[ix, iy+1], ins))
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

Limits a head iterate, after each linear solve of the head ([`Picard_iteration!`](@ref))
and on the extrapolated guess of a step ([`limit_head_guess!`](@ref)), as the [`PicardGuard`](@ref) `guard` says (a no-op
when it is inactive). Every limit protects a kink of the head rows, so each is switched on by what makes the kink, not by
the K-face scheme. A converged head is a fixed point of the limited iteration, so the solution is unchanged (within the
Picard tolerance). In order:

**Filled cap** (`filled_cap`, [`ConsistentGapCoupling`](@ref)): the step of a filled `GROUNDED` cell (any cell of
[`FilledCavities`](@ref) with [`NoUplift`](@ref); otherwise one whose head lies between the bed and the overburden head)
is capped at `|dN| <= max(|N|/2, po/4)`, and at least 1 m of head. On thick ice the gap update closes a thin gap within one
step for most `N`, so at `N` near `po` it is nearly flat in `N` and the Newton step of a filled row overshoots by
kilometres. Without the cap the iteration jumped between the bed and the overburden head every iteration (synthetic slab
under 1.5 to 3 km of ice, dt = 1 h: 500 of 500 iterations from step 5, and the water budget off by 10 to 100 times the
melt; with the cap 3 to 16 iterations at dt = 1 h and 1 day), and a cold start of the dry floor diverged at the first step
(Greenland and Antarctica 16 km, Thwaites 2 km at `dt = 1 h`): the first linear solve pushes the melt through the floor's
conductance, the head overshoots past the overburden, the gap sits at `b_max`, opening no longer takes up the melt, and the
dissipation melt feeds itself to NaN. With the cap every step converged on all three at `dt = 1 h` (30 days), `6 h` and
`24 h` (90 days), in at most about 30 iterations. Relative to `N`, the cap moves the high-suction cells (`N` up to
thousands of times `po`) geometrically: half the overburden head instead left the first step of Antarctica and Thwaites
unconverged after 500 iterations. Tested: `|N|/4` to `3|N|/4` and `po/10` to `po/4` behave alike; `po/2` near `N = 0`
fails on Thwaites at `dt = 24 h`. On the negative side (`N < 0`, head above the overburden, which a filled thick-ice cell
without uplift can reach only transiently) the relative cap would let `N` run away geometrically (`-5e6`, `-6e7`,
`-6.5e8`, ...), so there `N` may fall at most `po/4` below `min(N_prev, 0)` per iteration: without that, the first raw
iterate of the stuck-cells fixture (`N` around `-1e16`) made the exponential branch of `relax_update` overflow. That
fixture's first two steps stay unconverged (500 iterations), since its solution needs `N` near `-1e9` Pa: a bound that
doubles while the iterate keeps falling reaches it in 96 iterations, but then leaves steps 2 to 6 unconverged where the
linear bound converges from step 3, and no `dt` helps (`check_convergence`, `run.jl`). This bound
applies on the filled sheet only (no cavity or uplift storage): the unfilled sheet without uplift reaches `N` near `-6e8`
on the same fixture in 66 iterations, and the linear bound left that step unconverged after 500. Under
[`ImplicitUpwind`](@ref) with the dry floor and no cap the head matrix went singular at step 1 of that fixture.
[`ExplicitGapCoupling`](@ref), whose creep term is steep in `N`, is left unchanged.

**Regime stops** (wherever `stor > 0`, [`UnfilledCavities`](@ref), or `stor_up > 0`, [`IceUplift`](@ref)): an iterate
that crosses a regime boundary (the bed `zb` between the unfilled and filled rows, the overburden head `h_o` between the
filled and uplifted rows) stops just inside the new regime. The head row's storage slope jumps at these boundaries, from
`1/dt` (an unfilled or uplifted cell, plus the Newton terms of its water depth) to the creep derivative of a filled cell
(orders of magnitude smaller, and zero at `N = 0`), so one Newton step from the filled side lands far into the other
regime, where the water depth hits its floor and the Newton terms vanish, and the next step lands far back: the iteration
cycles (seen on SHW12 at dt = 1 h). From a point at the boundary the row is convex on the unfilled and uplifted side and
concave on the filled side, so Newton then approaches the root without crossing back. The same remedy as the "chopping"
of phase transitions in fully implicit reservoir simulators. Inside the unfilled and uplifted regimes the step is limited
too, as reservoir simulators limit the saturation change per Newton iteration: an uplifted cell's water depth at most
doubles per iteration, and the head of an unfilled cell sinks below the bed at most twice as deep per iteration, or by 3/4
of the gap if that is more (the gap empties in two iterations; below `zb - b` the water depth is at its floor and the head
of a dry cell reaches its value, which can be tens of metres below the bed, in about fifteen iterations). The factor 2
costs no time against 4 on Drang Drung (dt = 1 h to 1 day) and gives the same solution; at 8 a dt = 1 day run fails (a
singular head matrix after a runaway step). Without it, a Newton step from a thin layer, where the conductance and its
Newton terms are small, can land kilometres away (seen on Drang Drung at dt = 1 h: one cell 1.1 km below the bed, then a
neighbour 67 km above the overburden, then overflow).

**Outflow stops** (`guard.outflow_land`, `guard.outflow_ocean`): an outflow-only face is a kink of the row like the bed,
so an iterate that crosses the boundary head stops at it (`stop_at_boundary_head`); under [`SheetTopPotential`](@ref)
at the head where the cell's water level `h + b` meets it.

**Oscillation damping** (whenever the guard is active): a step that reverses the previous iteration's step of the same
cell is at most half of it, as in reservoir simulators. Near a root Newton's steps shrink much faster than that and the
damping never binds; it binds only where the iteration cycles, at a kink of the row (the bed, the overburden head, an
outflow face, or a face whose upwind donor flips), and there it brackets the root and halves the bracket each iteration,
as bisection does (seen on Drang Drung at dt = 24 h: a cell just below the bed whose Newton step from W = b empties three
quarters of the gap, and from there refills it, every iteration; without it Greenland's dry floor does not converge). The
previous steps (`s.dh_it`) are zeroed at the start of each step.
"""
limit_head_step!(guard::PicardGuard, s::State, h_prev, p::ModelParameters; filled_cap::Bool = guard.filled_cap) =
    guard.active ? (@parallel limit_head_step_kernel!(s.h, h_prev, s.dh_it, s.b, s.zb, s.po, s.stor, s.stor_up, p.rho_w, p.g, filled_cap, s.mask, guard.outflow_land, guard.outflow_ocean, s.db_x, s.db_y); s) : s

# Per GROUNDED cell: the head row's end-of-step water minus the water the fluxes and sources leave, W + dt (mdot/rho_w
# + ieb - div q) less the change of its englacial water (e_v > 0, from the head h_old at the start of the step), as
# update_water_budget! takes it (water_converged, elliptic_solver.jl). The row's water is that of
# compute_b_w_iterate! but with the current melt rate: s.b_w takes the gap's melt opening from the start of the step (mdot0).
# With `relative`, divided by the sum of the magnitudes of the cell's terms, the scale of its round-off.
@parallel_indices (ix, iy) function water_budget_residual_kernel!(r, relative, mask, Wo, q_x, q_y, mdot, ieb, dx, dy, dt, rho_w, b, h, zb, po, N, stor, stor_up, b_min, b_max, ggrav, gc, beta, abs_ub, A_visc, rho_i, n_minus_1, bump, b_next, h_old, e_v)
    if ix <= size(r, 1) && iy <= size(r, 2)
        if mask[ix, iy] == GROUNDED
            div = (q_x[ix+1, iy] - q_x[ix, iy]) / dx + (q_y[ix, iy+1] - q_y[ix, iy]) / dy
            W, _ = iterate_water(iterate_gap(gc, bump[ix, iy], b_next[ix, iy], zero(dt), false), b[ix, iy], Wo[ix, iy], h[ix, iy], zb[ix, iy], po[ix, iy], N[ix, iy], stor[ix, iy], stor_up[ix, iy],
                                 b_min, b_max, rho_w, ggrav, mdot[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], rho_i, n_minus_1)
            db_e = englacial_water(e_v, h[ix, iy], zb[ix, iy], po[ix, iy], stor[ix, iy], stor_up[ix, iy], rho_w, ggrav) -
                 englacial_water(e_v, h_old[ix, iy], zb[ix, iy], po[ix, iy], stor[ix, iy], stor_up[ix, iy], rho_w, ggrav)
            res = W + db_e - (Wo[ix, iy] + dt * (mdot[ix, iy] / rho_w + ieb[ix, iy] - div))
            scale = W + abs(db_e) + abs(Wo[ix, iy]) + dt * (abs(mdot[ix, iy] / rho_w) + abs(ieb[ix, iy]) + (abs(q_x[ix+1, iy]) + abs(q_x[ix, iy])) / dx + (abs(q_y[ix, iy+1]) + abs(q_y[ix, iy])) / dy)
            r[ix, iy] = relative ? res / (scale + floatmin(scale)) : res
        else
            r[ix, iy] = zero(dt)
        end
    end
    return
end

@parallel_indices (ix, iy) function update_water_budget_kernel!(W, mask, b, q_x, q_y, mdot, ieb, dx, dy, dt, rho_w, h, h_old, zb, po, stor, stor_up, e_v, ggrav)
    if ix <= size(W, 1) && iy <= size(W, 2)
        if mask[ix, iy] == GROUNDED
            div = (q_x[ix+1, iy] - q_x[ix, iy]) / dx + (q_y[ix, iy+1] - q_y[ix, iy]) / dy
            db_e = englacial_water(e_v, h[ix, iy], zb[ix, iy], po[ix, iy], stor[ix, iy], stor_up[ix, iy], rho_w, ggrav) -
                 englacial_water(e_v, h_old[ix, iy], zb[ix, iy], po[ix, iy], stor[ix, iy], stor_up[ix, iy], rho_w, ggrav)
            W[ix, iy] = max(zero(dt), W[ix, iy] + dt * (mdot[ix, iy] / rho_w + ieb[ix, iy] - div) - db_e)
        else
            W[ix, iy] = b[ix, iy]
        end
    end
    return
end

"""
$(TYPEDSIGNATURES)

Advances the water `s.W` by one step from the water budget on `GROUNDED` cells, `W = max(0, W + dt*(mdot/rho_w + ieb -
div(q)) - db_e)` with the step's converged fluxes (what the solve actually delivered, so the budget closes for every cell and
every gap scheme), `db_e` the change of the englacial water over the step (`p.e_v > 0`: `e_v (h - zb)` between the bed and
the overburden head, from the head `h_old` at its start to `s.h`; see `englacial_storage_terms`), and sets it to the gap `b`
elsewhere. Where the gap update and the head equation disagree (the
implicit gap schemes change the gap by a different amount than the head equation assumed, a held `b_max`/`b_min`, ...)
the difference stays in `W - b`: an empty part of the gap where `W < b`, water above the gap where `W > b` (the uplift
layer where the ice is lifted, see [`IceUplift`](@ref); otherwise released as a source the next step). The floor
`W >= 0` is the only water it can still create; a `Simulation` reports it per step ([`water_created`](@ref)). Called by [`step_b!`](@ref) under
[`UnfilledCavities`](@ref) (`water_is_budgeted`), right after the gap update and before `compute_beta!` reads the
uplift layer.
"""
update_water_budget!(s::State, g::Grid, p::ModelParameters, dt, h_old = s.h) =
    (@parallel update_water_budget_kernel!(s.W, s.mask, s.b, s.q_x, s.q_y, s.mdot, s.ieb, g.dx, g.dy, dt, p.rho_w, s.h, h_old, s.zb, s.po, s.stor, s.stor_up, p.e_v, p.g); s)

"""
$(TYPEDSIGNATURES)

The water budget of a [`Simulation`](@ref)'s steps (`sim.wb`), for every sheet: per step, the water the step created
on `GROUNDED` cells beyond what its sources and fluxes deliver, `created = sum(W_new + db_e - (W_old + dt*(mdot/rho_w +
ieb - div(q)))) * dx*dy` [m^3] (`db_e` the change of the englacial water, `p.e_v > 0`), and that step's grounded input
`dt*sum(mdot/rho_w + ieb) * dx*dy` [m^3]. Over all steps since the `Simulation` was built (kept in checkpoints):
`total_created` (net), `gross_created` and `gross_lost` (summed over cells and steps by sign, so water created in one
cell or step and lost in another does not cancel; `gross_lost <= 0`), `total_input`, and `unconverged`, the number of
steps taken with an unconverged head solve. Read them with [`water_created`](@ref).

Under [`UnfilledCavities`](@ref) (the default sheet) `created` is the water the floor `W >= 0` of
[`update_water_budget!`](@ref) creates, about the Picard tolerance times the input once the steps converge. Under
[`FilledCavities`](@ref) it is what the head rows and the gap update disagree on: about the Picard tolerance under
[`ConsistentGapCoupling`](@ref), percents of the input or more under [`ExplicitGapCoupling`](@ref). A negative value is
water lost. Mask changes are not part of it (their water is the `discarded_b` of [`set_mask!`](@ref)/`update_frozen_bed!`).
"""
struct WaterBudget{F <: AbstractFloat, A <: AbstractArray{F}}
    expected::A # per cell, the water the budget leaves, W_old + dt*(mdot/rho_w + ieb - div q) - db_e, then the step's excess over it
    src::A      # per cell, the step's input dt*(mdot/rho_w + ieb)
    created::Base.RefValue{F}
    input::Base.RefValue{F}
    total_created::Base.RefValue{F}
    total_input::Base.RefValue{F}
    gross_created::Base.RefValue{F}
    gross_lost::Base.RefValue{F}
    unconverged::Base.RefValue{Int}
end
WaterBudget(s::State) = (z = zero(eltype(s.W)); WaterBudget(similar(s.W), similar(s.W), Ref(z), Ref(z), Ref(z), Ref(z), Ref(z), Ref(z), Ref(0)))

# The budget's running totals, by name: what a checkpoint keeps.
const WATER_BUDGET_TOTALS = (:total_created, :total_input, :gross_created, :gross_lost, :unconverged)

# Before the gap update: expected <- the budget's water, src <- the grounded input (both per cell, zero off GROUNDED).
@parallel_indices (ix, iy) function expected_water_kernel!(Wx, src, mask, W, q_x, q_y, mdot, ieb, dx, dy, dt, rho_w, h, h_old, zb, po, stor, stor_up, e_v, ggrav)
    if ix <= size(Wx, 1) && iy <= size(Wx, 2)
        if mask[ix, iy] == GROUNDED
            div = (q_x[ix+1, iy] - q_x[ix, iy]) / dx + (q_y[ix, iy+1] - q_y[ix, iy]) / dy
            db_e = englacial_water(e_v, h[ix, iy], zb[ix, iy], po[ix, iy], stor[ix, iy], stor_up[ix, iy], rho_w, ggrav) -
                 englacial_water(e_v, h_old[ix, iy], zb[ix, iy], po[ix, iy], stor[ix, iy], stor_up[ix, iy], rho_w, ggrav)
            src[ix, iy] = dt * (mdot[ix, iy] / rho_w + ieb[ix, iy])
            Wx[ix, iy] = W[ix, iy] + src[ix, iy] - dt * div - db_e
        else
            src[ix, iy] = zero(dt)
            Wx[ix, iy] = zero(dt)
        end
    end
    return
end

# After the water update: expected <- W - expected on GROUNDED cells, the water the step created there.
@parallel_indices (ix, iy) function excess_water_kernel!(Wx, mask, W)
    if ix <= size(Wx, 1) && iy <= size(Wx, 2)
        Wx[ix, iy] = mask[ix, iy] == GROUNDED ? W[ix, iy] - Wx[ix, iy] : zero(eltype(Wx))
    end
    return
end

"""
$(TYPEDSIGNATURES)

The water budget of `sim`'s last step and of all its steps (see [`WaterBudget`](@ref)), in m^3: `(created, input,
total_created, total_input, gross_created, gross_lost, unconverged)`, the last a count of steps. `created/input` is the
fraction of the step's grounded input the step created.
"""
water_created(wb::WaterBudget) = (created = wb.created[], input = wb.input[], total_created = wb.total_created[], total_input = wb.total_input[],
                                  gross_created = wb.gross_created[], gross_lost = wb.gross_lost[], unconverged = wb.unconverged[])

# The water a GROUNDED cell's head row stored under the sheets whose water is not budgeted (water_is_budgeted, run.jl):
# the new gap less the empty depth below the bed (UnfilledCavities, budgeted in a Simulation), plus the uplift layer above
# the overburden head (IceUplift). Off GROUNDED the gap.
@parallel_indices (ix, iy) function water_from_head_kernel!(W, mask, b, h, zb, po, rho_wg, unfilled, uplift)
    if ix <= size(W, 1) && iy <= size(W, 2)
        if mask[ix, iy] == GROUNDED
            z = zero(eltype(W))
            empty = unfilled ? max(z, zb[ix, iy] - h[ix, iy]) : z
            lifted = uplift ? max(z, h[ix, iy] - (zb[ix, iy] + po[ix, iy] / rho_wg)) : z
            W[ix, iy] = max(b[ix, iy] - empty, z) + lifted
        else
            W[ix, iy] = b[ix, iy]
        end
    end
    return
end

"""
$(TYPEDSIGNATURES)

Sets the water `s.W` from the new gap and the solved head, for the sheets whose water is not budgeted: `W = max(b -
max(0, zb - h), 0) + max(0, h - h_o)` on `GROUNDED` cells, the first term under [`UnfilledCavities`](@ref) (whose
water a [`Simulation`](@ref) takes from the budget instead, [`update_water_budget!`](@ref)) and the second under [`IceUplift`](@ref), `h_o = zb +
po/(rho_w g)`; `W = b` elsewhere and under SHAKTI's original sheet. Under ConsistentGapCoupling this is the water the
head row stores, which closes the budget to the Picard tolerance once the step converges.
"""
water_from_head!(s::State, p::ModelParameters, cf::AbstractCavityFilling, up) =
    (@parallel water_from_head_kernel!(s.W, s.mask, s.b, s.h, s.zb, s.po, p.rho_w * p.g, cf isa UnfilledCavities, up isa IceUplift); s) # up: the AbstractIceUplift (ice_uplift.jl, included later)

@parallel_indices (ix, iy) function constrain_water_kernel!(W, b, unfilled, full, dry_floor, b_min)
    if ix <= size(W, 1) && iy <= size(W, 2)
        z = zero(eltype(W))
        w = max(W[ix, iy], z)
        bb = b[ix, iy]
        W[ix, iy] = unfilled  ? w :
                    dry_floor ? (bb > b_min ? bb : min(w, bb)) :
                    full      ? bb : max(w, bb)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Brings the water `s.W` within what the simulation's sheet allows, before a step and after a setup or mask change:
`W >= 0` everywhere; under [`FilledCavities`](@ref) the gap is full, `W >= b`, and `W = b` without
[`IceUplift`](@ref); under the dry floor (`dry_floor`) `W = b` above `b_min` and `0 <= W <= b` at it. Idempotent on the
water the sheet's own steps leave; it acts on a `State` that ran under other options (a spin-up, a checkpoint) and on
cells a mask change reseeded empty ([`thaw_cells!`](@ref)), which hold `b_min` of water under the filled sheet without
the dry floor.
"""
constrain_water!(s::State, p::ModelParameters, cf::AbstractCavityFilling, up, dry_floor::Bool) =
    (@parallel constrain_water_kernel!(s.W, s.b, cf isa UnfilledCavities, up isa NoUplift, dry_floor, p.b_min); s)

"""
$(TYPEDSIGNATURES)

The empty part of the gap [m], `max(0, b - W)` per cell: an unfilled cavity's empty volume, or the dry volume of the
dry floor (`dry_floor`). A derived diagnostic, also reachable as the observable `"b_empty"` ([`get_observable`](@ref)).
"""
empty_gap(s::State) = max.(s.b .- s.W, zero(eltype(s.W)))

"""
$(TYPEDSIGNATURES)

The water above the gap [m], `max(0, W - b)` per cell: the uplift layer under [`IceUplift`](@ref) where the ice is
lifted (see `uplift_layer`, `gap_height.jl`), otherwise water the budget keeps above the gap. A derived diagnostic, also
reachable as the observable `"b_uplift"` ([`get_observable`](@ref)).
"""
water_above_gap(s::State) = max.(s.W .- s.b, zero(eltype(s.W)))
