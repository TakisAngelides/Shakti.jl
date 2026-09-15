"""
$(TYPEDSIGNATURES)

Advances `state.h` by one timestep under the parabolic head scheme (`p.e_v != 0`): a single
backward-Euler linear solve ([`solve_parabolic_linear_system!`](@ref), via `ls`), then refreshes
every field that depends on the new `h` via [`refresh_head_dependents!`](@ref) (`elliptic_solver.jl`)
so the next timestep's solve is consistent.

Every nonlinear coefficient (`K`, `N`, `A_visc`, `b`, `mdot`, `beta`, `abs_ub`) going into that one
linear solve is lagged at whatever `state` held when this is called, with no correction -- unlike
[`Parabolic_iteration!`](@ref)/[`ParabolicPicardSolver`](@ref) below, this never re-evaluates them
against the new `h`. That lag is provably harmless once the state has actually reached a fixed
point (nothing left to lag behind), which is what [`parabolic_solver_test.jl`](@ref)'s "steady
state matches EllipticHeadScheme's" test exercises; away from a fixed point (large `dt`, or a
locally stiff cell -- e.g. a thin, marine-grounded cell near flotation) it can leave real error
in the transient uncorrected for the rest of the run, since nothing here re-solves with fresher
coefficients before moving on to the next timestep. Kept as the plain, non-iterating building
block (still used standalone in `parabolic_solver_test.jl`); [`ParabolicHeadScheme`](@ref)'s own
`step_h!` uses [`Parabolic_loop!`](@ref) instead specifically to correct for this.

# Notes

`state`/`grid`/`p`/`mt` are taken as separate arguments rather than a bundled `sim::Simulation`,
same reasoning as [`elliptic_solver!`](@ref): this file is included before `simulation.jl`, so
`ParabolicHeadScheme{PPS}` can use a proper `PPS <: ParabolicPicardSolver` bound.
"""
function parabolic_solver!(ls::AbstractLinearSolver, state::State, grid::Grid, p::ModelParameters, mt::MeltTerms, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw, dt; cnc::AbstractCellNClamping = NoCellNClamping(), ds::AbstractDiffusionScheme = NoDiffusion())

    s, g = state, grid

    # s.h itself is a valid h_old here: nothing has touched it yet this call, so "current s.h"
    # and "s.h at the start of this timestep" genuinely coincide -- unlike Parabolic_iteration!
    # below, where the same read would be a bug (see update_SALS_parabolic_kernel!'s docstring).
    solve_parabolic_linear_system!(ls, s, g, p, kfs, dt, s.h, ds) # update the h field, including the -div(D*grad(b)) source term under WithDiffusion

    refresh_head_dependents!(s, g, p, mt, kfs, sl; cnc, ds)

end

"""
$(TYPEDSIGNATURES)

Drives a Picard-style iteration of the backward-Euler parabolic head solve to nonlinear
convergence *within one real timestep* -- the parabolic counterpart of
[`PicardSolver`](@ref)/[`Picard_loop!`](@ref), same fields, same convergence check (relative
max-norm of the head update). Necessary because a single [`parabolic_solver!`](@ref) call lags
every nonlinear coefficient (`K`, `N`, `A_visc`, `b`, `mdot`, `beta`) at the *start-of-step*
state -- fine exactly at a fixed point (the lag vanishes there), but a real, demonstrated source
of error away from one: at large `dt`, a locally stiff cell (thin, marine-grounded ice near
flotation) can develop a transient `N<0` excursion under a single lagged solve that the elliptic
scheme does not show at the same `dt`. Repeating the backward-Euler solve with `K`/`A_visc`/`b`/
`mdot`/`beta` refreshed from the previous iterate drives that lag to zero, the same way Picard
iteration refreshes the elliptic scheme's own lagged coefficients -- at `iters=1` this reduces
exactly to a single [`parabolic_solver!`](@ref) call (the old, uncorrected behavior), so raising
`iters` only ever adds correction, never changes what a converged answer is.

This alone turned out not to be enough on the real Greenland grid: plain fixed-point refresh of
every coefficient (no Newton term anywhere) failed to converge at all on the same stiff cells
(500/500 iterations, `N<0` still growing every step) -- confirmed empirically, not just in theory.
`update_SALS_parabolic_kernel!`/`update_MFLS_parabolic_kernel!` (`linear_solver.jl`) therefore also
carry the *same* Newton linearization of the creep-closure term's own `N`-dependence that the
elliptic kernels use (baked into `aP`/`rhs`, not something this loop does) -- `K`/`A_visc`/`b`/
`mdot`/`beta` are still plain-Picard-refreshed across iterations here, only the creep term's `N`
sensitivity gets the Newton treatment, but that was the term actually responsible for the
non-convergence.

Build one with the [`ParabolicPicardSolver(iters, tol, ls, g; ...)`](@ref) convenience constructor
below, not this positional one directly; `converged`/`last_iter` are updated in place by
[`Parabolic_loop!`](@ref) each time it's called.
"""
mutable struct ParabolicPicardSolver{F <: AbstractFloat, LS <: AbstractLinearSolver, HR <: AbstractHeadRelaxation, A <: AbstractArray}
    iters::Int # how many within-timestep iterations to do at most
    tol::F # tolerance for stopping the loop
    ls::LS # linear solver
    converged::Bool # whether the loop converged before iters was exhausted
    last_iter::Int # at which iteration the loop stopped at any given time step
    hr::HR # head relaxation
    h_old::A      # head at the START of this real timestep, captured once by Parabolic_loop! before its first sub-iteration and held fixed for all of them -- the backward-Euler storage term's "old" value. NOT the same thing as h_prev below (see update_SALS_parabolic_kernel!'s docstring for why conflating the two breaks convergence)
    h_prev::A     # previous-SUB-ITERATION head (changes every sub-iteration), for the convergence check and under-relaxation
    delta_h::A    # change in head between iterations, for the convergence check
    check_every::Int # see PicardSolver's own field of the same name for the reasoning
end

"""
$(TYPEDSIGNATURES)

Builds a [`ParabolicPicardSolver`](@ref) with up to `iters` within-timestep iterations, relative
tolerance `tol`, linear solver `ls`, on grid `g`. `alpha` (in `(0, 1]`) enables
[`UnderHeadRelaxation`](@ref) if given, otherwise [`NoHeadRelaxation`](@ref) is used -- same
keywords, same meaning as [`PicardSolver`](@ref)'s own constructor. Pass `iters = 1` to recover
the old, non-iterating behavior exactly (a single lagged-coefficient solve per real timestep).
"""
function ParabolicPicardSolver(iters, tol, ls::AbstractLinearSolver, g::Grid; alpha = nothing, check_every::Int = DEFAULT_CHECK_EVERY)

    if alpha === nothing
        hr = NoHeadRelaxation()
    else
        hr = UnderHeadRelaxation(floattype(alpha))
    end

    h_old   = initialize_center_field(g)
    h_prev  = initialize_center_field(g)
    delta_h = initialize_center_field(g)

    return ParabolicPicardSolver(iters, floattype(tol), ls, false, 0, hr, h_old, h_prev, delta_h, check_every)

end

"""
$(TYPEDSIGNATURES)

One backward-Euler iteration of the parabolic head equation: solves the linear system for a new
`h` ([`solve_parabolic_linear_system!`](@ref), at timestep `dt`, against the fixed `h_old`),
optionally relaxes it ([`relax_h!`](@ref)), then refreshes every field that depends on the new `h`
via [`refresh_head_dependents!`](@ref) (`elliptic_solver.jl`) so the next iteration's coefficients
are consistent -- the exact parabolic counterpart of [`Picard_iteration!`](@ref).

`h_old` (fixed for the whole real timestep) and `h_prev` (the previous sub-iteration's head,
which does change every call) are deliberately two different arguments -- see
[`update_SALS_parabolic_kernel!`](@ref)'s docstring (`linear_solver.jl`) for why passing the same
array for both breaks convergence.
"""
function Parabolic_iteration!(ls::AbstractLinearSolver, hr::AbstractHeadRelaxation, s::State, g::Grid, p::ModelParameters, mt::MeltTerms, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw, dt, h_old, h_prev; cnc::AbstractCellNClamping = NoCellNClamping(), ds::AbstractDiffusionScheme = NoDiffusion())

    solve_parabolic_linear_system!(ls, s, g, p, kfs, dt, h_old, ds)
    relax_h!(hr, s, h_prev)

    refresh_head_dependents!(s, g, p, mt, kfs, sl; cnc, ds)

end

"""
$(TYPEDSIGNATURES)

Repeatedly calls [`Parabolic_iteration!`](@ref) (up to `pps.iters` times, at timestep `dt`),
checking convergence every `pps.check_every` iterations via a relative max-norm on the head update
(`max|delta_h| / (max|h| + eps) < pps.tol`), and sets `pps.converged`/`pps.last_iter` accordingly
-- the exact parabolic counterpart of [`Picard_loop!`](@ref). Captures `pps.h_old` from `s.h`
exactly once, before the first sub-iteration -- every sub-iteration solves backward-Euler from
that same fixed starting point, not from whatever the previous sub-iteration happened to leave in
`s.h` (see [`update_SALS_parabolic_kernel!`](@ref)'s docstring for why that distinction is not
just bookkeeping: conflating the two is a real bug that prevented this loop from converging at
all on stiff cells, confirmed on the real Greenland grid).
"""
function Parabolic_loop!(pps::ParabolicPicardSolver, state::State, grid::Grid, p::ModelParameters, mt::MeltTerms, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw, dt; cnc::AbstractCellNClamping = NoCellNClamping(), ds::AbstractDiffusionScheme = NoDiffusion())

    s = state

    @. pps.h_old = s.h # fixed for the whole timestep, captured before any sub-iteration mutates s.h

    pps.converged = false
    pps.last_iter = 0

    @inbounds for iter in 1:pps.iters

        @. pps.h_prev = s.h

        Parabolic_iteration!(pps.ls, pps.hr, state, grid, p, mt, kfs, sl, dt, pps.h_old, pps.h_prev; cnc, ds)

        @. pps.delta_h = s.h - pps.h_prev

        if iter % pps.check_every == 0 || iter == pps.iters
            delta_h_max = mapreduce(abs, max, pps.delta_h; init = zero(eltype(s.h)))
            h_max = mapreduce(abs, max, s.h; init = zero(eltype(s.h)))
            if delta_h_max / (h_max + eps(eltype(s.h))) < pps.tol
                pps.converged = true
                pps.last_iter = iter
                return
            end
        end

    end

    pps.last_iter = pps.iters
    return

end
