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
end

function PicardSolver(iters, tol, ls::AbstractLinearSolver, g::Grid; alpha = nothing)

    if alpha === nothing
        hr = NoHeadRelaxation()
    else
        hr = UnderHeadRelaxation(floattype(alpha))
    end

    h_prev  = initialize_center_field(g)
    delta_h = initialize_center_field(g)

    return PicardSolver(iters, floattype(tol), ls, false, 0, hr, h_prev, delta_h)
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
        @. ps.h_prev = s.h

        Picard_iteration!(ps.ls, ps.hr, state, grid, p, shs, kfs, mi, ps.h_prev)

        @. ps.delta_h = s.h - ps.h_prev
        if maximum(abs, ps.delta_h) / (norm(s.h, Inf) + eps(eltype(s.h))) < ps.tol
            ps.converged = true
            ps.last_iter = iter
            return
        end

    end

    ps.last_iter = ps.iters
    return

end

function Picard_iteration!(ls::AbstractLinearSolver, hr::AbstractHeadRelaxation, s::State, g::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, mi::AbstractMeltInput, h_prev)

    solve_linear_system!(ls, s, g, p, kfs, mi)
    relax_h!(hr, s, h_prev) # damp the raw Picard update before anything downstream of h is recomputed, so the next iteration's coefficients are consistent with the relaxed h

    # Update state variables that depend on the new h
    compute_dhdx!(s, g)
    compute_dhdy!(s, g)

    compute_pw!(s, p)
    compute_dpwdx!(s, g) # feeds compute_sensible!'s sensible-heat term (via compute_mdot! below)
    compute_dpwdy!(s, g)
    compute_N!(s)

    compute_q_x!(s, p)
    compute_q_y!(s, p)

    compute_Re_x!(s, p)
    compute_Re_y!(s, p)
    compute_Re!(s)

    compute_taub_x!(s, p)
    compute_taub_y!(s, p)

    compute_mdot!(s, p, shs)

    compute_K!(s, p)

end

