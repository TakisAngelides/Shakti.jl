"""
$(TYPEDSIGNATURES)

Whether/how the raw Picard update to `state.h` is damped before the next iteration -- multiple
dispatch on the concrete subtype ([`NoHeadRelaxation`](@ref)/[`UnderHeadRelaxation`](@ref)) picks
whether [`relax_h!`](@ref) is a no-op or an under-relaxation blend with the previous iteration's
head.
"""
abstract type AbstractHeadRelaxation end

"""
$(TYPEDSIGNATURES)

No damping: each Picard iteration's raw linear-solve result is used as-is.
"""
struct NoHeadRelaxation <: AbstractHeadRelaxation end

"""
$(TYPEDSIGNATURES)

Under-relaxation: blends the new head with the previous iteration's head by `alpha` (see
[`relax_h!`](@ref)), trading slower convergence for extra stability on stiff problems.
"""
struct UnderHeadRelaxation{F <: AbstractFloat}  <: AbstractHeadRelaxation
    alpha::F
end

"""
$(TYPEDSIGNATURES)

No-op under [`NoHeadRelaxation`](@ref).
"""
relax_h!(::NoHeadRelaxation, state::State, h_prev) = state

"""
$(TYPEDSIGNATURES)

Blends `state.h` with `h_prev` by `hr.alpha`: `h = alpha*h + (1-alpha)*h_prev`. Applied before
anything downstream of `h` is recomputed, so the next iteration's coefficients are consistent
with the relaxed `h`. Alpha should be between 0 and 1 and usually taken closer to 0 than to 1.
"""
function relax_h!(hr::UnderHeadRelaxation, state::State, h_prev)
    alpha = hr.alpha
    @. state.h = alpha * state.h + (1 - alpha) * h_prev
    return state
end

"""
$(TYPEDSIGNATURES)

Drives the Picard iteration used to solve the nonlinear elliptic equation for hydraulic head:
holds the linear solver (`ls`), optional head relaxation (`hr`), iteration/tolerance settings,
and the scratch fields (`h_prev`, `delta_h`) the convergence check needs. Build one with the
keyword-free constructor below; `converged`/`last_iter` are updated in place by
[`Picard_loop!`](@ref) each time it's called. It is a mutable struct to be able to change the 
iter and converged fields, but also gives the flexibility to be changing the linear solver along the 
simulation if desired.
"""
mutable struct PicardSolver{F <: AbstractFloat, LS <: AbstractLinearSolver, HR <: AbstractHeadRelaxation, A <: AbstractArray}
    iters::Int # how many Picard iterations to do for a Picard loop
    tol::F # tolerance for stopping the Picard loop
    ls::LS # linear solver
    converged::Bool # whether the Picard loop converged
    last_iter::Int # at which iteration the Picard loop stopped at any given time step
    hr::HR # head relaxation
    h_prev::A     # previous-iteration head, for the Picard convergence check and under-relaxation
    delta_h::A    # change in head between iterations, for the Picard convergence check
    check_every::Int # convergence check forces a GPU->CPU sync (the reduction result has to reach the host for the `if`); only check every this many iterations rather than every one, trading a few possible extra (cheap, async) Picard iterations for fewer syncs -- see below
end

# Measured (both Threads and Metal, 32x32) check_every=1 having equal-or-lower
# total Picard iterations AND lower wall time than check_every=3 or 10: checking
# every iteration lets Picard stop as soon as it's actually converged, instead of
# running up to check_every-1 extra iterations past convergence before noticing.
# On Threads there's no sync to amortize in the first place, so this isn't a
# surprise; on Metal the sync-avoidance benefit check_every was designed for
# didn't show up either, at least not at this (small) grid size -- a larger grid,
# where each iteration does enough real work to make the sync proportionally
# cheaper, might tip this the other way.
"""
Default value of [`PicardSolver`](@ref)'s `check_every`: `1` (check convergence every
iteration).
"""
const DEFAULT_CHECK_EVERY = 1

"""
$(TYPEDSIGNATURES)

Builds a [`PicardSolver`](@ref) with up to `iters` iterations, relative tolerance `tol`, linear
solver `ls`, on grid `g`. `alpha` (in `(0, 1]`) enables [`UnderHeadRelaxation`](@ref) if given,
otherwise [`NoHeadRelaxation`](@ref) is used.
"""
function PicardSolver(iters, tol, ls::AbstractLinearSolver, g::Grid; alpha = nothing, check_every::Int = DEFAULT_CHECK_EVERY)

    if alpha === nothing
        hr = NoHeadRelaxation()
    else
        hr = UnderHeadRelaxation(floattype(alpha))
    end

    h_prev  = initialize_center_field(g)
    delta_h = initialize_center_field(g)

    return PicardSolver(iters, floattype(tol), ls, false, 0, hr, h_prev, delta_h, check_every)
end

"""
$(TYPEDSIGNATURES)

Solves the nonlinear elliptic equation for hydraulic head at the current timestep via Picard
iteration ([`Picard_loop!`](@ref)), updating `state` and `ps` (`ps.converged`/`ps.last_iter`) in
place.

# Notes

`state`/`grid`/`p`/`shs` are taken as separate arguments (rather than a bundled `sim::Simulation`)
so this file doesn't need `Simulation` to already be defined -- it can be included, and
`PicardSolver`'s struct fully written, before `simulation.jl`, letting `EllipticHeadScheme{PS}`
(`simulation.jl`) use a proper `PS <: PicardSolver` bound instead of leaving `PS` unbounded. The `shs`
which stands for sensible heat scheme chooses between including or not including the last term of Eq. 7
in https://doi.org/10.1017/jog.2023.39; a melt rate term accounting for the changes in the pressure-melting-point 
temperature with changes in water pressure. The `kfs` that stands for K face scheme determines how to calculate the
transmissivity on a grid cell face given the two cell center values, with choices such as arithmetic or harmonic mean.
The `sl` sliding law determines which sliding law to use to calculate the basal shear stress tau_b. The choices can be
regularized Coulomb law, linear law, or prescribed by the user.
"""
function elliptic_solver!(ps::PicardSolver, state::State, grid::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw)
    Picard_loop!(ps, state, grid, p, shs, kfs, sl)
end

"""
$(TYPEDSIGNATURES)

Repeatedly calls [`Picard_iteration!`](@ref) (up to `ps.iters` times), checking convergence every
`ps.check_every` iterations via a relative max-norm on the head update
(`max|delta_h| / (max|h| + eps) < ps.tol`), and sets `ps.converged`/`ps.last_iter` accordingly.
"""
function Picard_loop!(ps::PicardSolver, state::State, grid::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw)

    s = state

    # Initialize PicardSolver state
    ps.converged = false
    ps.last_iter = 0

    @inbounds for iter in 1:ps.iters # start the Picard loop

        # Store previous head for convergence check
        @. ps.h_prev = s.h

        Picard_iteration!(ps.ls, ps.hr, state, grid, p, shs, kfs, sl, ps.h_prev) # run one linear solve to update h and the relevant fields

        @. ps.delta_h = s.h - ps.h_prev

        if iter % ps.check_every == 0 || iter == ps.iters
            # The convergence check uses the maximum difference between h and h_prev normalized by the maximum value of h to compare to the tolerance and stop the Picard loop if reached
            # Both maximum(abs, delta_h) and norm(h, Inf) (== maximum(abs, h)) computed in one fused reduction pass instead of two separate ones -- halves the GPU->CPU syncs per check.
            # note: mapreduce(f, op, A, B), zips A and B elementwise instead of requiring one array
            delta_h_max, h_max = mapreduce(
                (dh, hh) -> (abs(dh), abs(hh)), # the map function
                (a, b) -> (max(a[1], b[1]), max(a[2], b[2])), # the reduction operation, a and b are each a tuple produced by the map step, we are comparing two dh values together a[1], b[1] and two h values together a[2], b[2] to get the max from each one
                ps.delta_h, s.h; # two arrays to zip and do the map reduction on
                init = (zero(eltype(s.h)), zero(eltype(s.h))) # start the accumulator for both slots
            )
            if delta_h_max / (h_max + eps(eltype(s.h))) < ps.tol
                ps.converged = true
                ps.last_iter = iter
                return
            end
        end

    end

    ps.last_iter = ps.iters
    return

end

"""
$(TYPEDSIGNATURES)

One Picard iteration: solves the linearized system for a new `h` ([`solve_elliptic_linear_system!`](@ref)),
optionally relaxes it ([`relax_h!`](@ref)), then refreshes every field that depends on the new `h`
(`pw`, `N`, `q`/`Re`, `taub`, `mdot`, `K`) so the next iteration's linearization is consistent. The
water depth `b` is left untouched within one Picard loop until we step out of it and update `b` following
Eq. 2 of https://gmd.copernicus.org/articles/11/2955/2018/.
"""
function Picard_iteration!(ls::AbstractLinearSolver, hr::AbstractHeadRelaxation, s::State, g::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw, h_prev)

    solve_elliptic_linear_system!(ls, s, g, p, kfs) # update the h field
    relax_h!(hr, s, h_prev) # update the h field again according to the relaxation parameter, damp the raw Picard update before anything downstream of h is recomputed, so the next iteration's coefficients are consistent with the relaxed h

    # Update state variables that depend on the new h
    compute_dhdxy!(s, g) # updates gradient of h in both x and y directions in one kernel to reduce the number of kernels

    compute_pw!(s, p) # update water pressure
    compute_dpwdxy!(s, g) # update the water pressure gradient in both x and y in one kernel, feeds compute_sensible!'s sensible-heat term (via compute_mdot! below)
    compute_N!(s) # update effective pressure (ice overburden pressure - pw)

    compute_q_and_Re_xy!(s, p) # update water flux qx, qy and Reynold's number on faces so Re_x, Re_y all in one kernel to reduce kernel - the Reynold's number is calculated based on the solution of the quadratic equation that defines it (Eq. 5 and 7 combined from https://gmd.copernicus.org/articles/11/2955/2018/)
    compute_Re!(s) # update the Reynold's number based on the Re_x and Re_y doing an average over the four faces of a grid cell

    compute_taub_xy!(s, p, sl) # update the basal shear stress based on the sliding law `sl` chosen

    compute_mdot!(s, p, shs) # update the melt rate based on whether we should include the sensible heat term or not which is determined by `shs`

    compute_K!(s, p) # update the transmissivity

end

