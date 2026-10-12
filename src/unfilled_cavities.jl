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
an empty cavity neither carries nor draws water. It is the water present at the start of the step,
not at the current head iterate: like `b` in the legacy equations it stays fixed during the Picard
loop. That lag makes the water-depth dependence of the flux explicit in time where the water depth is
free (an unfilled or uplifted cell): a step longer than the time such a cell takes to fill or drain,
about `dx*W/(3|q|)` (minutes at 100 m for a thin free-surface layer), overshoots, and the solution
oscillates from step to step or Picard cycles. Taking the depth at the head iterate in a plain Picard
iteration does not help: the one-cell Picard map then has slope ~ -3(W + dzb)/W, and neither Anderson
acceleration nor [`NewtonJFNKSolver`](@ref) converged on the SHW12 benchmark at dt = 6 h. What does is
the end-of-step depth together with the Newton terms of its dependence on the head,
[`ImplicitUpwind`](@ref) ([`compute_b_w_iterate!`](@ref)).
"""
compute_b_w!(s::State, p::ModelParameters) = (@parallel compute_b_w_kernel!(s.b_w, s.b, s.b_empty, p.b_min, s.b_uplift); s)

# The gap at the end of the step that the head rows assume (linear_solver.jl): the start-of-step gap under
# ExplicitGapCoupling (the original scheme, which lags the gap in the conductance), the gap scheme's update G(N) under
# ConsistentGapCoupling (SchemeGapUpdate), with its slope in the head dG/dh = -rho_w g dG/dN (central difference, as in
# scheme_head_terms) where N follows the head (a filled cell).
@inline end_of_step_gap(::ExplicitGapCoupling, b, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) = b
@inline end_of_step_gap(gu::SchemeGapUpdate, b, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1) =
    scheme_gap(gu, b, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1)
@inline end_of_step_gap_slope(::ExplicitGapCoupling, b, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav) = zero(b)
@inline function end_of_step_gap_slope(gu::SchemeGapUpdate, b, b_min, b_max, mdot, beta, abs_ub, A, Nv, rho_i, n_minus_1, rho_w, ggrav)
    δ = 1e-6 * max(abs(Nv), oftype(Nv, 1e3))
    Gm = scheme_gap(gu, b, b_min, b_max, mdot, beta, abs_ub, A, Nv - δ, rho_i, n_minus_1)
    Gp = scheme_gap(gu, b, b_min, b_max, mdot, beta, abs_ub, A, Nv + δ, rho_i, n_minus_1)
    return rho_w * ggrav * max(zero(b), (Gm - Gp) / (2δ))
end

@parallel_indices (ix, iy) function compute_b_w_iterate_kernel!(b_w, dbw_dh, b, h, zb, po, N, stor, stor_up, b_min, b_max, rho_w, ggrav, gc, mdot, beta, abs_ub, A_visc, rho_i, n_minus_1)
    if ix <= size(b_w, 1) && iy <= size(b_w, 2)
        bb = b[ix, iy]
        x  = h[ix, iy] - zb[ix, iy]                # pressure head
        xo = po[ix, iy] / (rho_w * ggrav)          # pressure head at the overburden
        if stor[ix, iy] > 0 && x < 0               # unfilled: the gap at N = po minus the empty depth zb - h
            W = end_of_step_gap(gc, bb, b_min, b_max, mdot[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], po[ix, iy], rho_i, n_minus_1) + x
            slope = one(bb)
        elseif stor_up[ix, iy] > 0 && x > xo       # uplifted: the gap at N = 0 plus the layer h - h_o
            W = end_of_step_gap(gc, bb, b_min, b_max, mdot[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], zero(bb), rho_i, n_minus_1) + (x - xo)
            slope = one(bb)
        else                                       # filled: the gap (the start-of-step one, or G(N) under ConsistentGapCoupling)
            Nv = N[ix, iy]
            W = end_of_step_gap(gc, bb, b_min, b_max, mdot[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], Nv, rho_i, n_minus_1)
            slope = end_of_step_gap_slope(gc, bb, b_min, b_max, mdot[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], Nv, rho_i, n_minus_1, rho_w, ggrav)
        end
        floor_w = min(bb, b_min)
        b_w[ix, iy]    = W > floor_w ? W : floor_w
        dbw_dh[ix, iy] = W > floor_w ? slope : zero(bb)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Water depth at the end of the step as a function of the current head iterate, for [`ImplicitUpwind`](@ref), and its slope
`s.dbw_dh = dW/dh`. With the gap `G` the head rows assume for the end of the step (`gc`): `W = G(p_o) - (zb - h)` where the
gap may be partly empty (`s.stor > 0`, [`UnfilledCavities`](@ref)) and `h < zb`, `W = G(0) + (h - h_o)` where the ice may be
lifted (`s.stor_up > 0`, [`IceUplift`](@ref)) and `h > h_o`, `W = G(N)` otherwise, never below `min(b, p.b_min)`. These are
the head rows' own water: the depth they conserve (with the melt opening of `G` from the start of the step,
[`store_start_of_step_melt!`](@ref)). Under [`ExplicitGapCoupling`](@ref) `G` is the start-of-step gap `b`
(the head rows add the gap change only to the water budget), so a filled cell's depth is `b` with slope 0 (the gap is lagged
in its conductance, as in the original scheme) and the unfilled and uplifted depths move one to one with the head. Under
[`ConsistentGapCoupling`](@ref) `G` is the gap scheme's own update: the conductance of every cell then uses the end-of-step
water depth, a filled cell's with slope `rho_w g |dG/dN|`, and the coupled head-gap step is implicit (no lag of the gap in
the conductance either). Both are continuous across the regimes and equal `b` exactly under [`FilledCavities`](@ref),
[`NoUplift`](@ref) and [`ExplicitGapCoupling`](@ref).
"""
compute_b_w_iterate!(s::State, p::ModelParameters, gc::AbstractGapCoupling = ExplicitGapCoupling()) =
    (@parallel compute_b_w_iterate_kernel!(s.b_w, s.dbw_dh, s.b, s.h, s.zb, s.po, s.N, s.stor, s.stor_up, p.b_min, p.b_max, p.rho_w, p.g, gc, s.mdot0, s.beta, s.abs_ub, s.A_visc, p.rho_i, p.n_minus_1_exp); s)

"""
$(TYPEDSIGNATURES)

Keeps the melt rate at the start of the step (`s.mdot0`), for the end-of-step gap in the conductance of
[`ImplicitUpwind`](@ref) ([`compute_b_w_iterate!`](@ref)); a no-op for every other K-face scheme. That gap takes its melt
opening `dt*mdot/rho_i` from the start of the step and only its creep closure from the head iterate: the dissipation melt
grows with the flux, so a melt opening at the iterate would close the loop `W -> K -> q -> mdot -> W` inside the Picard
iteration, a positive feedback (the growth of a channel) with a gain of order `dt*(3/W)*mdot/rho_i`, which the iteration
cannot follow at large `dt` (seen on a 3 m^3/s moulin at dt = 3 h). The head rows keep the melt at the iterate, so the water
balance and its consistency with the gap update are unaffected; only the conductance lags the melt by one step, as the
original scheme lags the whole gap.
"""
store_start_of_step_melt!(::AbstractKFaceScheme, s::State) = s
store_start_of_step_melt!(::ImplicitUpwind, s::State) = (copyto!(s.mdot0, s.mdot); s)

@parallel_indices (ix, iy) function limit_regime_crossing_kernel!(h, h_prev, dh_it, b, zb, po, stor, stor_up, rho_w, ggrav)
    if ix <= size(h, 1) && iy <= size(h, 2)
        hn, hp, z = h[ix, iy], h_prev[ix, iy], zb[ix, iy]
        ins = max(1e-6 * b[ix, iy], 64 * eps(abs(z) + one(z))) # a step inside the new regime, keeping W ~ b
        bc = b[ix, iy]
        if stor[ix, iy] > 0
            hp >= z && hn < z && (hn = z - ins) # filled -> unfilled: stop just below the bed (the unfilled row next)
            hp < z && hn > z && (hn = z)        # unfilled -> filled: stop at the bed (the filled row next)
            hp < z && (hn = max(hn, hp - max(3 * bc / 4, 3 * (z - hp)))) # within unfilled: the head's depth below the bed grows at most 4-fold per iteration, or by 3/4 of the gap
        end
        if stor_up[ix, iy] > 0
            ho = z + po[ix, iy] / (rho_w * ggrav)
            hp <= ho && hn > ho && (hn = ho + ins) # filled -> uplifted
            hp > ho && hn < ho && (hn = ho)        # uplifted -> filled
            hp > ho && (hn = min(hn, hp + 3 * (bc + hp - ho))) # within uplifted: the water depth grows at most 4-fold per iteration
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
simulators limit the saturation change per Newton iteration: an uplifted cell's water depth grows at most 4-fold per
iteration, and the head of an unfilled cell sinks below the bed at most 4-fold per iteration, or by 3/4 of the gap if that
is more (the gap empties in two iterations; below `zb - b` the water depth is at its floor and the head of a dry cell
reaches its value, which can be tens of metres below the bed, in about ten iterations instead of one step of 3/4 of the gap
at a time). Without it, a
Newton step from a thin layer, where the conductance and its Newton terms are small, can land kilometres away (seen on
Drang Drung at dt = 1 h: one cell 1.1 km below the bed, then a neighbour 67 km above the overburden, then overflow). A
converged head is a fixed point of the limited iteration too, so the solution is unchanged. Last, a step that reverses
the previous iteration's step of the same cell is at most half of it (the "oscillation damping" of reservoir simulators):
near a root Newton's steps shrink much faster than that and the damping never binds; it binds only where the iteration
cycles, at a kink of the row (the bed or the overburden head, where the slope of the water depth jumps, or a face whose
upwind donor flips), and there it brackets the root and halves the bracket each iteration, as bisection does (seen on
Drang Drung at dt = 24 h: a cell just below the bed whose Newton step from W = b empties three quarters of the gap, and
from there refills it, every iteration). The previous steps (`s.dh_it`) are zeroed at the start of each step.
"""
limit_regime_crossing!(::AbstractKFaceScheme, s::State, h_prev, p::ModelParameters) = s
limit_regime_crossing!(::ImplicitUpwind, s::State, h_prev, p::ModelParameters) =
    (@parallel limit_regime_crossing_kernel!(s.h, h_prev, s.dh_it, s.b, s.zb, s.po, s.stor, s.stor_up, p.rho_w, p.g); s)

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
