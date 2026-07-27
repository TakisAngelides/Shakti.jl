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
with the relaxed `h`.
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
[`Picard_loop!`](@ref) each time it's called.
"""
mutable struct PicardSolver{F <: AbstractFloat, LS <: AbstractLinearSolver, HR <: AbstractHeadRelaxation, A <: AbstractArray}
    iters::Int
    tol::F
    ls::LS
    converged::Bool
    last_iter::Int
    hr::HR
    h_prev::A     # previous-iteration head, for the Picard convergence check and under-relaxation
    delta_h::A    # change in head between iterations, for the Picard convergence check
    check_every::Int # convergence check forces a GPU->CPU sync (the reduction result
                      # has to reach the host for the `if`); only check every this many
                      # iterations rather than every one, trading a few possible extra
                      # (cheap, async) Picard iterations for fewer syncs -- see below
end

# Measured (both Threads and Metal, 32x32) check_every=1 having equal-or-lower
# total Picard iterations AND lower wall time than check_every=3 or 10: checking
# every iteration lets Picard stop as soon as it's actually converged, instead of
# running up to check_every-1 extra iterations past convergence before noticing.
# On Threads there's no sync to amortize in the first place, so this isn't a
# surprise; on Metal the sync-avoidance benefit check_every was designed for
# didn't show up either, at least not at this (small) grid size -- a larger grid,
# where each iteration does enough real work to make the sync proportionally
# cheaper, might tip this the other way, but untested for now. Revisit if that
# changes.
"""
Default value of [`PicardSolver`](@ref)'s `check_every`: `1` (check convergence every
iteration). See the module-level note above for the benchmarking behind this choice.
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
(`simulation.jl`) use a proper `PS <: PicardSolver` bound instead of leaving `PS` unbounded.
"""
function elliptic_solver!(ps::PicardSolver, state::State, grid::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, mi::AbstractMeltInput, sl::AbstractSlidingLaw)
    Picard_loop!(ps, state, grid, p, shs, kfs, mi, sl)
end

"""
$(TYPEDSIGNATURES)

Repeatedly calls [`Picard_iteration!`](@ref) (up to `ps.iters` times), checking convergence every
`ps.check_every` iterations via a relative max-norm on the head update
(`max|delta_h| / (max|h| + eps) < ps.tol`), and sets `ps.converged`/`ps.last_iter` accordingly.
"""
function Picard_loop!(ps::PicardSolver, state::State, grid::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, mi::AbstractMeltInput, sl::AbstractSlidingLaw)

    s = state

    # Initialize PicardSolver state
    ps.converged = false
    ps.last_iter = 0

    @inbounds for iter in 1:ps.iters

        # Store previous head for convergence check
        @. ps.h_prev = s.h

        Picard_iteration!(ps.ls, ps.hr, state, grid, p, shs, kfs, mi, sl, ps.h_prev)

        @. ps.delta_h = s.h - ps.h_prev

        if iter % ps.check_every == 0 || iter == ps.iters
            # Both maximum(abs, delta_h) and norm(h, Inf) (== maximum(abs, h)) computed
            # in one fused reduction pass instead of two separate ones -- halves the
            # GPU->CPU syncs per check (see check_every above for why syncs matter).
            delta_h_max, h_max = mapreduce(
                (dh, hh) -> (abs(dh), abs(hh)),
                (a, b) -> (max(a[1], b[1]), max(a[2], b[2])),
                ps.delta_h, s.h;
                init = (zero(eltype(s.h)), zero(eltype(s.h)))
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

One Picard iteration: solves the linearized system for a new `h` ([`solve_linear_system!`](@ref)),
optionally relaxes it ([`relax_h!`](@ref)), then refreshes every field that depends on the new `h`
(`pw`, `N`, `q`/`Re`, `taub`, `mdot`, `K`) so the next iteration's linearization is consistent.
"""
function Picard_iteration!(ls::AbstractLinearSolver, hr::AbstractHeadRelaxation, s::State, g::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, mi::AbstractMeltInput, sl::AbstractSlidingLaw, h_prev)

    solve_linear_system!(ls, s, g, p, kfs, mi)
    relax_h!(hr, s, h_prev) # damp the raw Picard update before anything downstream of h is recomputed, so the next iteration's coefficients are consistent with the relaxed h

    # Update state variables that depend on the new h
    compute_dhdxy!(s, g)

    compute_pw!(s, p)
    compute_dpwdxy!(s, g) # feeds compute_sensible!'s sensible-heat term (via compute_mdot! below)
    compute_N!(s)

    compute_q_and_Re_xy!(s, p) # exact, lag-free q/Re solve -- see water_flux.jl's note
    compute_Re!(s)

    compute_taub_xy!(s, p, sl)

    compute_mdot!(s, p, shs)

    compute_K!(s, p)

end

