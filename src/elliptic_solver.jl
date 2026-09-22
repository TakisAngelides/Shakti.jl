"""
$(TYPEDSIGNATURES)

Whether/how the raw Picard update to `state.h` is damped (or extrapolated) before the next
iteration -- multiple dispatch on the concrete subtype ([`NoHeadRelaxation`](@ref)/
[`UnderHeadRelaxation`](@ref)/[`AndersonHeadRelaxation`](@ref)) picks whether [`relax_h!`](@ref) is
a no-op, an under-relaxation blend with the previous iteration's head, or an Anderson-accelerated
extrapolation using several previous iterations' history.
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

Anderson acceleration (depth `depth`, mixing/damping `beta`) of the Picard fixed-point map
`h -> solve_elliptic_linear_system!(...)`. Instead of blending the new head with just the
immediately preceding iterate (as [`UnderHeadRelaxation`](@ref) does, with a fixed weight), Anderson
acceleration keeps the last `depth` (iterate, residual) pairs and, each iteration, solves a small
(at most `depth`x`depth`) least-squares problem for the linear combination of past residuals that
comes closest to canceling the current one, then extrapolates the next iterate using those same
weights. This is the standard "Type-I" formulation (residual-difference least squares -- Walker &
Ni, SIAM J. Numer. Anal. 49(4), 1715-1735, 2011; Fang & Saad, Numer. Linear Algebra Appl. 16(3),
2009), reformulated in terms of increments (`ΔX`/`ΔF`) rather than the equivalent
constrained-weights form, since increments avoid needing to explicitly enforce `sum(alpha) = 1`.

`beta = 1` recovers "pure" Anderson extrapolation (no extra damping beyond what the least-squares
step itself provides); `beta < 1` mixes in some of [`UnderHeadRelaxation`](@ref)'s damping on top,
which may help stability on stiff problems at the cost of some acceleration.

Uses the normal equations (`(ΔF'ΔF) \\ (ΔF'f_k)`) rather than a QR-based least-squares solve, for
simplicity -- the well-known less-robust choice (normal equations square `ΔF`'s condition number,
a real risk once consecutive residual differences become nearly collinear, typically once the
residual is already small) but acceptable for the small `depth` tested here, since a `depth x
depth` solve is negligible either way next to the O(n) linear solve it's accelerating. A
QR-with-column-dropping implementation would be needed for a fully robust production version; as a
cheap stopgap, `gamma_clip` catches the specific failure mode actually observed empirically
(`linear_solver_benchmarks.tex`/`beyond_solver_choice.tex`: depth 5-8 sometimes extrapolated in the
wrong direction, making convergence *worse* than plain Picard) by discarding the extrapolation and
falling back to a plain beta-damped step whenever the solved weights blow up
(`maximum(abs, gamma) > gamma_clip`) -- a solved weight of that magnitude means the normal-equations
solve is numerically unreliable for this iteration, and the safe thing is to skip the extrapolation
for just that one iteration rather than apply a wild, unreliable correction to the head field.

History is reset once per Picard loop (see [`reset_relaxation!`](@ref)/[`Picard_loop!`](@ref)):
different timesteps solve a different fixed-point map (the Newton-linearization point and boundary
data both change), so carrying history across timesteps would extrapolate against a stale map.
"""
mutable struct AndersonHeadRelaxation{F <: AbstractFloat, M <: AbstractMatrix{F}, V <: AbstractVector{F}} <: AbstractHeadRelaxation
    depth::Int
    beta::F
    gamma_clip::F # discard the extrapolation and fall back to a plain step if maximum(abs, gamma) exceeds this
    Xhist::M   # n x (depth+1): past x_k (pre-relaxation iterates), oldest first, newest in the last column
    Fhist::M   # n x (depth+1): past f_k = g(x_k) - x_k, same column convention as Xhist
    dX::M      # n x depth workspace: consecutive columns of Xhist, differenced
    dF::M      # n x depth workspace: consecutive columns of Fhist, differenced
    gram::Matrix{F}  # depth x depth workspace for dF'dF
    rhs::V     # depth workspace for dF'f_k
    gamma::V   # depth workspace, the solved least-squares weights
    fk::V      # n workspace for the current residual g(x_k) - x_k
    count::Int # how many valid raw (x, f) columns are currently held, 0..depth+1
end

"""
$(TYPEDSIGNATURES)

Builds an [`AndersonHeadRelaxation`](@ref) for grid `g`. Defaults (`depth=3`, `beta=1.0`) are the
empirically best-performing setting found for Shakti's Picard loop
(`beyond_solver_choice.tex`: 18.5% fewer Picard iterations and 15.3% less wall time than plain
Picard at 256x256 with `CholeskyDirectSolver`, combined with `-t 8`) -- depth 5 and 8 were
tested and are both *worse* than depth 3, non-monotonically, so raising `depth` "for more history"
is not a safe assumption here. `gamma_clip` (default `10.0`) bounds the fallback-triggering
threshold described in [`AndersonHeadRelaxation`](@ref)'s docstring.
"""
function AndersonHeadRelaxation(g::Grid{F}; depth::Int = 3, beta = 1.0, gamma_clip = 10.0) where F
    n = g.nx * g.ny
    Xhist = @zeros(n, depth + 1)
    Fhist = @zeros(n, depth + 1)
    dX    = @zeros(n, depth)
    dF    = @zeros(n, depth)
    gram  = zeros(F, depth, depth)
    rhs   = zeros(F, depth)
    gamma = zeros(F, depth)
    fk    = @zeros(n)
    return AndersonHeadRelaxation(depth, F(beta), F(gamma_clip), Xhist, Fhist, dX, dF, gram, rhs, gamma, fk, 0)
end

"""
$(TYPEDSIGNATURES)

No-op for relaxation schemes with no history to reset (everything except
[`AndersonHeadRelaxation`](@ref)).
"""
reset_relaxation!(::AbstractHeadRelaxation) = nothing

"""
$(TYPEDSIGNATURES)

Clears [`AndersonHeadRelaxation`](@ref)'s history -- called once at the start of every
[`Picard_loop!`](@ref) (i.e. once per timestep), since a new timestep means a new fixed-point map
and old (iterate, residual) pairs are no longer valid extrapolation data for it.
"""
reset_relaxation!(hr::AndersonHeadRelaxation) = (hr.count = 0; nothing)

"""
$(TYPEDSIGNATURES)

Anderson-accelerated update: see [`AndersonHeadRelaxation`](@ref) for the derivation. `state.h`
holds `g(x_k)` (the raw, just-solved Picard update) on entry; `h_prev` is `x_k`.
"""
function relax_h!(hr::AndersonHeadRelaxation, state::State, h_prev)

    depth = hr.depth
    hvec = vec(state.h)    # g(x_k) on entry (read below into hr.fk); reused in place as the write-target
                            # for the new iterate further down, since it's a reshape (no copy) of state.h --
                            # writing through hvec IS writing into state.h, just with a flat (n,) shape that
                            # matches hprev_v/hr.fk/dX/dF for broadcasting and mul!.
    hprev_v = vec(h_prev)  # x_k

    @. hr.fk = hvec - hprev_v # f_k = g(x_k) - x_k, computed before hvec (aliasing state.h) is overwritten below

    # Shift history one column left (drop the oldest), append (x_k, f_k) as the newest column.
    @views hr.Xhist[:, 1:depth] .= hr.Xhist[:, 2:depth+1]
    @views hr.Fhist[:, 1:depth] .= hr.Fhist[:, 2:depth+1]
    hr.Xhist[:, depth+1] .= hprev_v
    hr.Fhist[:, depth+1] .= hr.fk
    hr.count = min(hr.count + 1, depth + 1)

    mk = min(hr.count - 1, depth) # usable DIFFERENCE columns (need >= 2 raw columns for 1 difference)

    if mk <= 0
        # First iteration of this Picard loop: no history yet, fall back to a plain beta-damped Picard step.
        @. hvec = hprev_v + hr.beta * hr.fk
        return state
    end

    lo = depth + 1 - mk # first raw column index among the mk+1 most recent ones
    @views begin
        X = hr.Xhist[:, lo:depth+1]
        F = hr.Fhist[:, lo:depth+1]
        dX = hr.dX[:, 1:mk]
        dF = hr.dF[:, 1:mk]
        for j in 1:mk
            @. dX[:, j] = X[:, j+1] - X[:, j]
            @. dF[:, j] = F[:, j+1] - F[:, j]
        end

        gram = hr.gram[1:mk, 1:mk]
        rhs  = hr.rhs[1:mk]
        mul!(gram, dF', dF)
        mul!(rhs, dF', hr.fk)
        gamma = hr.gamma[1:mk]
        gamma .= gram \ rhs # small mk x mk dense solve -- negligible next to the O(n) elliptic solve this accelerates

        if maximum(abs, gamma) > hr.gamma_clip
            # Normal-equations solve is numerically unreliable this iteration (near-collinear
            # ΔF columns) -- discard the extrapolation rather than risk moving h in the wrong
            # direction; fall back to a plain beta-damped step for just this one iteration.
            @. hvec = hprev_v + hr.beta * hr.fk
            return state
        end

        @. hvec = hprev_v + hr.beta * hr.fk
        mul!(hvec, dX, gamma, -1.0, 1.0)
        mul!(hvec, dF, gamma, -hr.beta, 1.0)
    end

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

`state`/`grid`/`p`/`mt` are taken as separate arguments (rather than a bundled `sim::Simulation`)
so this file doesn't need `Simulation` to already be defined -- it can be included, and
`PicardSolver`'s struct fully written, before `simulation.jl`, letting `EllipticHeadScheme{PS}`
(`simulation.jl`) use a proper `PS <: PicardSolver` bound instead of leaving `PS` unbounded. The `mt`
(a [`MeltTerms`](@ref)) chooses which terms of Eq. 7 in https://doi.org/10.1017/jog.2023.39 to include
in the melt rate -- e.g. its `Sensible` flag is the last term, accounting for changes in the
pressure-melting-point temperature with changes in water pressure. The `kfs` that stands for K face scheme determines how to calculate the
transmissivity on a grid cell face given the two cell center values, with choices such as arithmetic or harmonic mean.
The `sl` sliding law determines which sliding law to use to calculate the basal shear stress tau_b. The choices can be
regularized Coulomb law, linear law, or prescribed by the user. The `ds` keyword (an
[`AbstractDiffusionScheme`](@ref), `linear_solver.jl`; [`NoDiffusion`](@ref) by default) chooses whether the
channel-wall diffusion coefficient `D` is refreshed every Picard iteration ([`WithDiffusion`](@ref) recomputes it
from the current `q`/`∇h`/`∇P_w`; `NoDiffusion` skips it entirely).
"""
function elliptic_solver!(ps::PicardSolver, state::State, grid::Grid, p::ModelParameters, mt::MeltTerms, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw; cnc::AbstractCellNClamping = NoCellNClamping(), ds::AbstractDiffusionScheme = NoDiffusion())
    Picard_loop!(ps, state, grid, p, mt, kfs, sl; cnc, ds)
end

"""
$(TYPEDSIGNATURES)

Repeatedly calls [`Picard_iteration!`](@ref) (up to `ps.iters` times), checking convergence every
`ps.check_every` iterations via a relative max-norm on the head update
(`max|delta_h| / (max|h| + eps) < ps.tol`), and sets `ps.converged`/`ps.last_iter` accordingly.
"""
function Picard_loop!(ps::PicardSolver, state::State, grid::Grid, p::ModelParameters, mt::MeltTerms, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw; cnc::AbstractCellNClamping = NoCellNClamping(), ds::AbstractDiffusionScheme = NoDiffusion())

    s = state

    # Initialize PicardSolver state
    ps.converged = false
    ps.last_iter = 0
    reset_relaxation!(ps.hr) # no-op except for AndersonHeadRelaxation, whose history is only valid within one timestep's fixed-point map

    @inbounds for iter in 1:ps.iters # start the Picard loop

        # Store previous head for convergence check
        @. ps.h_prev = s.h

        Picard_iteration!(ps.ls, ps.hr, state, grid, p, mt, kfs, sl, ps.h_prev; cnc, ds) # run one linear solve to update h and the relevant fields

        @. ps.delta_h = s.h - ps.h_prev

        if iter % ps.check_every == 0 || iter == ps.iters
            # The convergence check uses the maximum difference between h and h_prev normalized by the maximum value of h to compare to the tolerance and stop the Picard loop if reached
            # Two separate single-array mapreduce calls, NOT one fused two-array mapreduce(f, op, A, B) as this used to be: mapreduce over a SINGLE array is Julia's genuinely non-allocating
            # streaming reduction, but the two-array form silently falls back to map+collect(zip(...)) internally, materializing a full temporary array of (delta_h, h) tuples every single
            # call -- confirmed via Profile.Allocs to allocate ~1.3MB/call at a ~200x400 grid, pure waste since check_every=1's own comment already establishes there's no GPU sync to
            # amortize on the Threads backend. NOT the cause of a separate, much larger long-run memory leak also found on this workload (traced instead to a Julia SparseArrays/CHOLMOD
            # ldiv! bug, JuliaSparse/SparseArrays.jl#726, unrelated to this call) -- this fix reduces allocation/GC pressure, nothing more.
            delta_h_max = mapreduce(abs, max, ps.delta_h; init = zero(eltype(s.h)))
            h_max = mapreduce(abs, max, s.h; init = zero(eltype(s.h)))
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

Refreshes every state field that depends on the just-solved `h` (`pw`, `N`, `q`/`Re`, `taub`,
`mdot`, `K`, and -- under [`WithDiffusion`](@ref) -- `D`) -- the tail shared by one elliptic Picard
iteration ([`Picard_iteration!`](@ref)) and one parabolic backward-Euler iteration
([`Parabolic_iteration!`](@ref), `parabolic_solver.jl`), once each has updated `h` by its own
linear solve. The water depth `b` is left untouched here in either case until we step out of the
head solve entirely and update `b` following Eq. 2 of
https://gmd.copernicus.org/articles/11/2955/2018/.
"""
function refresh_head_dependents!(s::State, g::Grid, p::ModelParameters, mt::MeltTerms, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw; cnc::AbstractCellNClamping = NoCellNClamping(), ds::AbstractDiffusionScheme = NoDiffusion())

    compute_dhdxy!(s, g) # updates gradient of h in both x and y directions in one kernel to reduce the number of kernels

    compute_pw!(s, p) # update water pressure
    compute_dpwdxy!(s, g) # update the water pressure gradient in both x and y in one kernel, feeds compute_sensible!'s sensible-heat term (via compute_mdot! below)
    compute_N!(s, p, cnc) # update effective pressure (ice overburden pressure - pw)

    compute_q_and_Re_xy!(s, p) # update water flux qx, qy and Reynold's number on faces so Re_x, Re_y all in one kernel to reduce kernel - the Reynold's number is calculated based on the solution of the quadratic equation that defines it (Eq. 5 and 7 combined from https://gmd.copernicus.org/articles/11/2955/2018/)
    compute_Re!(s) # update the Reynold's number based on the Re_x and Re_y doing an average over the four faces of a grid cell

    compute_taub_xy!(s, p, sl) # update the basal shear stress based on the sliding law `sl` chosen

    compute_mdot!(s, p, mt) # update the melt rate, including/excluding each term per `mt`

    compute_K!(s, p) # update the transmissivity

    compute_D!(s, p, mt, ds) # update the channel-wall diffusion coefficient (no-op under NoDiffusion)

    return s

end

"""
$(TYPEDSIGNATURES)

One Picard iteration: solves the linearized system for a new `h` ([`solve_elliptic_linear_system!`](@ref)),
optionally relaxes it ([`relax_h!`](@ref)), then refreshes every field that depends on the new `h`
via [`refresh_head_dependents!`](@ref) so the next iteration's linearization is consistent.
"""
function Picard_iteration!(ls::AbstractLinearSolver, hr::AbstractHeadRelaxation, s::State, g::Grid, p::ModelParameters, mt::MeltTerms, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw, h_prev; cnc::AbstractCellNClamping = NoCellNClamping(), ds::AbstractDiffusionScheme = NoDiffusion())

    solve_elliptic_linear_system!(ls, s, g, p, kfs, ds) # update the h field, including the -div(D*grad(b)) source term under WithDiffusion
    relax_h!(hr, s, h_prev) # update the h field again according to the relaxation parameter, damp the raw Picard update before anything downstream of h is recomputed, so the next iteration's coefficients are consistent with the relaxed h

    refresh_head_dependents!(s, g, p, mt, kfs, sl; cnc, ds)

end

