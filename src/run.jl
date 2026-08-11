"""
$(TYPEDSIGNATURES)

Runs `sim` from `t=0` (or from a checkpoint, see `restart_path`) up to `sim.tsteps`, calling
[`step!`](@ref) each iteration and recording output via `sim.observer`.

# Notes

`checkpoint_every`/`checkpoint_path`: if both given, `sim.state` is saved to `checkpoint_path`
(overwriting the previous checkpoint -- only the latest is kept) every `checkpoint_every` tsteps,
so a run killed at any point can be resumed via `restart_path` below instead of starting over from
`t=0`.

`restart_path`: if given, resumes from a checkpoint written by a *previous* `run!` call instead of
starting fresh at `t=0` -- loads `sim.state`/`sim.total_time` and continues the loop from the
checkpointed tstep + 1 up to `sim.tsteps` (still the *original* total step count, not "how many
more steps to do"). The observer's output file is reopened (not truncated) via `resume!` so it's
appended to across the restart; see `observer.jl` for how each file writer handles a tstep that
the crashed run may have already written.
"""
function run!(sim::Simulation; checkpoint_every::Union{Nothing, Int} = nothing, checkpoint_path::Union{Nothing, String} = nothing, restart_path::Union{Nothing, String} = nothing)

    if (checkpoint_every === nothing) != (checkpoint_path === nothing)
        error("checkpoint_every and checkpoint_path must be given together")
    end

    if restart_path === nothing
        sim.total_time[] = zero(sim.dt) # reset so the same Simulation can be run! more than once, e.g. chained runs sharing one state
        prepare!(sim.observer, sim.state)
        observe!(sim.observer, sim.state, 0, sim.total_time[])
        start_t = 0
    else
        start_t = load_checkpoint!(sim, restart_path)
        resume!(sim.observer, sim.state, start_t)
    end

    for t in (start_t + 1):sim.tsteps

        step_time = @elapsed step!(sim)

        sim.total_time[] += sim.dt

        observe!(sim.observer, sim.state, t, sim.total_time[])

        if checkpoint_every !== nothing && t % checkpoint_every == 0
            save_checkpoint(checkpoint_path, sim, t)
        end

        if sim.verbose
            converged, last_iter = picard_status(sim.hs)
            println("$t / $(sim.tsteps) completed in $(round(step_time; digits = 4))s. Picard converged: $converged in $last_iter iterations")
            if converged === false
                s = sim.state
                println("  diagnostics: N=$(extrema(Array(s.N))) Re=$(extrema(Array(s.Re))) b=$(extrema(Array(s.b))) h=$(extrema(Array(s.h)))")
            end
            flush(stdout) # println alone doesn't reach the log file promptly under sbatch: stdout is fully block-buffered (not line-buffered) once it's redirected to a file rather than a terminal
        end

    end

    finalize!(sim.observer, sim.state)

end

"""
$(TYPEDSIGNATURES)

Returns `(converged, last_iter)` for `hs`'s Picard solve at the current timestep -- dispatched
(rather than an `isa` check) so this stays correct if another `AbstractHeadScheme` is ever added:
`EllipticHeadScheme` has a `PicardSolver` to report on, `ParabolicHeadScheme` doesn't (it's a
single backward-Euler solve per timestep, not an iterative one -- see `step_h!` below).
"""
picard_status(hs::EllipticHeadScheme) = (hs.ps.converged, hs.ps.last_iter)
picard_status(hs::ParabolicHeadScheme) = (missing, missing)

"""
$(TYPEDSIGNATURES)

Advances `sim` by one timestep: refreshes the melt input ([`update_ieb!`](@ref)), solves for the
new head ([`step_h!`](@ref)), then evolves the gap height ([`step_b!`](@ref)).
"""
function step!(sim::Simulation)

    update_ieb!(sim.mi, sim.state, sim.total_time[]) # no-op for ConstantMeltInput; rescales state.ieb for e.g. SeasonalMeltInput -- done once per timestep, before step_h!, since ieb only feeds the head equation (step_b! never reads it)
    step_h!(sim.hs, sim)
    step_b!(sim)

end

"""
$(TYPEDSIGNATURES)

Solves for the new hydraulic head under [`EllipticHeadScheme`](@ref): runs the Picard/elliptic
solve (`sim.mi`'s melt input for the current time was already refreshed by [`step!`](@ref)).
"""
function step_h!(hs::EllipticHeadScheme, sim::Simulation)
    elliptic_solver!(hs.ps, sim.state, sim.grid, sim.p, sim.shs, sim.kfs, sim.sl)
end

"""
$(TYPEDSIGNATURES)

Solves for the new hydraulic head under [`ParabolicHeadScheme`](@ref): a single backward-Euler
linear solve ([`parabolic_solver!`](@ref)), no Picard loop.
"""
function step_h!(hs::ParabolicHeadScheme, sim::Simulation)
    parabolic_solver!(hs.ls, sim.state, sim.grid, sim.p, sim.shs, sim.kfs, sim.sl, sim.dt)
end

"""
$(TYPEDSIGNATURES)

Evolves the gap height `sim.state.b` by one timestep, dispatching on `sim.gs`
(`ImplicitGapScheme()`/`ExplicitGapScheme()`) to [`compute_b!(sim, sim.gs)`](@ref) below.

# Notes

Only evolves `b` where hydrology is actually being solved (`GROUNDED`). Cells with a
Dirichlet-prescribed `pw` (`LAND`/`OCEAN`) or a frozen `h` (`OTHER_BASIN`) don't have a
meaningfully-evolving `b` in this model, so their `b` is simply left untouched at whatever it was
initialized to.
"""
compute_b!(sim::Simulation) = compute_b!(sim, sim.gs)

"""
$(TYPEDSIGNATURES)

Implicit (backward-Euler) update of `sim.state.b`: implicit on the creep closure term only, the scheme used by default.
"""
function compute_b!(sim::Simulation, ::ImplicitGapScheme)
    s, p = sim.state, sim.p
    @parallel compute_b_implicit_kernel!(s.b, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt, p.b_min, p.b_max)
    return sim
end

"""
$(TYPEDSIGNATURES)

Explicit (forward-Euler) update of `sim.state.b`: cheaper per step, but only stable for small enough `sim.dt`.
"""
function compute_b!(sim::Simulation, ::ExplicitGapScheme)
    s, p = sim.state, sim.p
    @parallel compute_b_explicit_kernel!(s.b, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt, p.b_min, p.b_max)
    return sim
end

"""
$(TYPEDSIGNATURES)

Evolves the gap height `sim.state.b` for one timestep ([`compute_b!`](@ref), dispatching
internally on `sim.gs`), then refreshes everything that depends on it (`beta`, `b_x`, `b_y`) so
they're ready for the *next* timestep's Picard loop.
"""
function step_b!(sim::Simulation)

    s, p = sim.state, sim.p

    compute_b!(sim)          # updates b based on the new state variables (GROUNDED cells only)

    compute_beta!(s, p, sim.oss) # opening-by-sliding parameter depends on the new b
    compute_b_x!(s)          # water depth on x faces
    compute_b_y!(s)          # water depth on y faces

end
