abstract type AbstractHeadRelaxation end

struct NoHeadRelaxation <: AbstractHeadRelaxation end
struct UnderHeadRelaxation{F <: AbstractFloat}  <: AbstractHeadRelaxation
    alpha::F
end

relax_h!(::NoHeadRelaxation, state::State, h_prev) = state

function relax_h!(hr::UnderHeadRelaxation, state::State, h_prev)
    alpha = hr.alpha
    @. state.h = alpha * state.h + (1 - alpha) * h_prev
    return state
end

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
const DEFAULT_CHECK_EVERY = 1

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

# state/grid/p/shs are taken as separate arguments (rather than a bundled
# sim::Simulation) so this file doesn't need Simulation to already be
# defined -- it can be included, and PicardSolver's struct fully written,
# before simulation.jl, letting EllipticHeadScheme{PS} (simulation.jl) use a
# proper PS <: PicardSolver bound instead of leaving PS unbounded.
function elliptic_solver!(ps::PicardSolver, state::State, grid::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, mi::AbstractMeltInput)
    Picard_loop!(ps, state, grid, p, shs, kfs, mi)
end

function Picard_loop!(ps::PicardSolver, state::State, grid::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, mi::AbstractMeltInput)

    s = state
    
    # Initialize PicardSolver state
    ps.converged = false
    ps.last_iter = 0

    @inbounds for iter in 1:ps.iters

        # Store previous head for convergence check
        @timeit PERF_TIMER "h_prev copy" @. ps.h_prev = s.h

        Picard_iteration!(ps.ls, ps.hr, state, grid, p, shs, kfs, mi, ps.h_prev)

        @timeit PERF_TIMER "delta_h" @. ps.delta_h = s.h - ps.h_prev

        if iter % ps.check_every == 0 || iter == ps.iters
            # Both maximum(abs, delta_h) and norm(h, Inf) (== maximum(abs, h)) computed
            # in one fused reduction pass instead of two separate ones -- halves the
            # GPU->CPU syncs per check (see check_every above for why syncs matter).
            delta_h_max, h_max = @timeit PERF_TIMER "convergence check" mapreduce(
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

function Picard_iteration!(ls::AbstractLinearSolver, hr::AbstractHeadRelaxation, s::State, g::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, mi::AbstractMeltInput, h_prev)

    @timeit PERF_TIMER "linear solve" solve_linear_system!(ls, s, g, p, kfs, mi)
    @timeit PERF_TIMER "relax_h" relax_h!(hr, s, h_prev) # damp the raw Picard update before anything downstream of h is recomputed, so the next iteration's coefficients are consistent with the relaxed h

    # Update state variables that depend on the new h
    @timeit PERF_TIMER "dhdx" compute_dhdx!(s, g)
    @timeit PERF_TIMER "dhdy" compute_dhdy!(s, g)

    @timeit PERF_TIMER "pw" compute_pw!(s, p)
    @timeit PERF_TIMER "dpwdx" compute_dpwdx!(s, g) # feeds compute_sensible!'s sensible-heat term (via compute_mdot! below)
    @timeit PERF_TIMER "dpwdy" compute_dpwdy!(s, g)
    @timeit PERF_TIMER "N" compute_N!(s)

    @timeit PERF_TIMER "q_x" compute_q_x!(s, p)
    @timeit PERF_TIMER "q_y" compute_q_y!(s, p)

    @timeit PERF_TIMER "Re_x" compute_Re_x!(s, p)
    @timeit PERF_TIMER "Re_y" compute_Re_y!(s, p)
    @timeit PERF_TIMER "Re" compute_Re!(s)

    @timeit PERF_TIMER "taub_x" compute_taub_x!(s, p)
    @timeit PERF_TIMER "taub_y" compute_taub_y!(s, p)

    @timeit PERF_TIMER "mdot" compute_mdot!(s, p, shs)

    @timeit PERF_TIMER "K" compute_K!(s, p)

end

