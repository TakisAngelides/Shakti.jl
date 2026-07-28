"""
$(TYPEDSIGNATURES)

Advances `state.h` by one timestep under the parabolic head scheme (`p.e_v != 0`): a single
backward-Euler linear solve ([`solve_parabolic_linear_system!`](@ref), via `ls`) instead of a
Picard loop, then refreshes every field that depends on the new `h` (`pw`, `N`, `q`/`Re`, `taub`,
`mdot`, `K`) so the next timestep's solve is consistent -- the same tail
[`Picard_iteration!`](@ref) runs after its own linear solve.

# Notes

`state`/`grid`/`p`/`shs` are taken as separate arguments rather than a bundled `sim::Simulation`,
same reasoning as [`elliptic_solver!`](@ref): this file is included before `simulation.jl`, so
`ParabolicHeadScheme{LS}` can use a proper `LS <: AbstractLinearSolver` bound.
"""
function parabolic_solver!(ls::AbstractLinearSolver, state::State, grid::Grid, p::ModelParameters, shs::AbstractSensibleHeatScheme, kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw, dt)

    s, g = state, grid

    solve_parabolic_linear_system!(ls, s, g, p, kfs, dt) # update the h field

    # Update state variables that depend on the new h -- same as Picard_iteration!'s tail
    compute_dhdxy!(s, g)

    compute_pw!(s, p)
    compute_dpwdxy!(s, g)
    compute_N!(s)

    compute_q_and_Re_xy!(s, p)
    compute_Re!(s)

    compute_taub_xy!(s, p, sl)

    compute_mdot!(s, p, shs)

    compute_K!(s, p)

end
