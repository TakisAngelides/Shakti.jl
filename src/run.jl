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

Under [`AdaptiveTimeStep`](@ref), the loop can also end *before* `sim.tsteps` if `total_time[]`
reaches `sim.ts.target_time` first (the intended way to stop a time-targeted adaptive run, since
the true step count needed to cover `target_time` isn't known in advance) -- `sim.tsteps` itself
should still be sized generously from `AdaptiveTimeStep`'s own `dt_min` (see its docstring) so the
loop is a real, finite upper bound either way, not just a number that happens to be big enough.
"""
function run!(sim::Simulation; checkpoint_every::Union{Nothing, Int} = nothing, checkpoint_path::Union{Nothing, String} = nothing, restart_path::Union{Nothing, String} = nothing)

    if (checkpoint_every === nothing) != (checkpoint_path === nothing)
        error("checkpoint_every and checkpoint_path must be given together")
    end

    if restart_path === nothing
        sim.total_time[] = zero(sim.dt[]) # reset so the same Simulation can be run! more than once, e.g. chained runs sharing one state
        prepare!(sim.observer, sim.state)
        observe!(sim.observer, sim.state, 0, sim.total_time[])
        start_t = 0
    else
        start_t = load_checkpoint!(sim, restart_path)
        resume!(sim.observer, sim.state, start_t)
    end

    for t in (start_t + 1):sim.tsteps

        step_time = @elapsed step!(sim)

        sim.total_time[] += sim.dt[]

        observe!(sim.observer, sim.state, t, sim.total_time[])

        if checkpoint_every !== nothing && t % checkpoint_every == 0
            save_checkpoint(checkpoint_path, sim, t)
        end

        if sim.verbose
            converged, last_iter = picard_status(sim.hs)
            println("$t / $(sim.tsteps) completed in $(round(step_time; digits = 4))s. Picard converged: $converged in $last_iter iterations")
            s = sim.state
            Narr = Array(s.N)
            if converged === false
                println("  diagnostics: N=$(extrema(Narr)) Re=$(extrema(Array(s.Re))) b=$(extrema(Array(s.b))) h=$(extrema(Array(s.h)))")
            end
            # N < 0 (water pressure exceeding ice overburden) is physically invalid under grounded
            # ice but isn't caught by Picard's own convergence check, so it can persist silently
            # even in a "converged" step -- report where/why it happens (not just that it does),
            # since it's the leading indicator of near-flotation cells destabilizing the solve.
            nmin, nmin_idx = findmin(Narr)
            if nmin < 0
                # Array(field)[idx] would copy the WHOLE domain just to read one scalar (this branch
                # fires on most steps once any cell goes sub-flotation, so that copy isn't free);
                # Array(field[ix:ix, iy:iy])[1] copies a single-element slice instead -- still
                # GPU-safe (no bare scalar getindex on a device array), just without the waste.
                ix, iy = Tuple(nmin_idx)
                at(field) = Array(field[ix:ix, iy:iy])[1]
                println("  N<0: min=$(round(nmin, sigdigits = 4)) Pa at $(Tuple(nmin_idx)) -- zb=$(round(at(s.zb), sigdigits = 4)) H=$(round(at(s.H), sigdigits = 4)) po=$(round(at(s.po), sigdigits = 4)) pw=$(round(at(s.pw), sigdigits = 4))")
            end
            flush(stdout) # println alone doesn't reach the log file promptly under sbatch: stdout is fully block-buffered (not line-buffered) once it's redirected to a file rather than a terminal
        end

        # AdaptiveTimeStep's target_time-based stop: sim.tsteps is still a real, finite upper
        # bound (sized from dt_min, see AdaptiveTimeStep's docstring), this is just the ordinary
        # "reached the target early" exit. target_time(::FixedTimeStep) is `nothing`, so this is a
        # no-op under the default scheme.
        (tt = target_time(sim.ts)) !== nothing && sim.total_time[] >= tt && break

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

Advances `sim` by one timestep: recomputes `sim.dt[]` if `sim.ts` is adaptive
([`update_dt!`](@ref)), refreshes the melt input ([`update_ieb!`](@ref)), solves for the new head
([`step_h!`](@ref)), then evolves the gap height ([`step_b!`](@ref)).
"""
function step!(sim::Simulation)

    update_dt!(sim) # no-op under FixedTimeStep; must run before step_h!/step_b! since both read sim.dt[]
    update_ieb!(sim.mi, sim.state, sim.total_time[]) # no-op for ConstantMeltInput; rescales state.ieb for e.g. SeasonalMeltInput -- done once per timestep, before step_h!, since ieb only feeds the head equation (step_b! never reads it)
    step_h!(sim.hs, sim)
    step_b!(sim)

end

"""
$(TYPEDSIGNATURES)

Recomputes `sim.dt[]` under [`AdaptiveTimeStep`](@ref); a no-op under [`FixedTimeStep`](@ref) (`dt`
never changes). Reads `sim.state`'s fields as they stand at the *start* of this step -- i.e. the
previous step's converged `N`/`u_b` -- into `ts.rate_field` via [`compute_dt_rate_kernel!`](@ref),
reduces that field to a single statistic ([`rate_statistic`](@ref), dispatched on
`AdaptiveTimeStep`'s `UsePercentile` type parameter), then sets
`sim.dt[] = clamp(safety_factor/stat, dt_min, dt_max)`, clipped further so the final step lands
exactly on `target_time` rather than overshooting it.
"""
update_dt!(sim::Simulation) = update_dt!(sim, sim.ts)
update_dt!(sim::Simulation, ::FixedTimeStep) = sim

function update_dt!(sim::Simulation, ts::AdaptiveTimeStep)
    s, p = sim.state, sim.p
    @parallel compute_dt_rate_kernel!(ts.rate_field, s.mask, s.b, s.abs_ub, s.A_visc, s.N, p.n_minus_1_exp, p.br, p.lr)
    stat = rate_statistic(ts.rate_field, ts)
    dt = stat > 0 ? ts.safety_factor / stat : ts.dt_max # stat<=0 means no grounded cell has a positive rate (e.g. all N<=0) -- fall back to dt_max rather than dividing by zero/a negative
    dt = clamp(dt, ts.dt_min, ts.dt_max)
    dt = min(dt, ts.target_time - sim.total_time[]) # clip the final step to land exactly on target_time instead of overshooting it
    sim.dt[] = dt
    return sim
end

"""
$(TYPEDSIGNATURES)

Reduces `rate` (this step's `C+gamma` field, see [`compute_dt_rate_kernel!`](@ref)) to the single
statistic [`update_dt!`](@ref) divides `safety_factor` by. `UsePercentile=false`: the plain domain
maximum -- a single GPU-native reduction, no allocation, fully conservative (governed by the
single fastest-relaxing cell). `UsePercentile=true`: one bulk `copyto!` of `rate` into
`ts.host_rate` (CUDA.jl/Metal.jl's real device->host transfer -- scalar-indexing `rate` directly
per grounded cell would be illegal/catastrophically slow on a GPU backend), then a plain CPU loop
gathers the `GROUNDED`-cell values (precomputed indices) into `ts.scratch`, sorts it in place, and
indexes the `ts.percentile`-th entry. None of this allocates -- every buffer is preallocated once
by [`AdaptiveTimeStep`](@ref)'s constructor.
"""
rate_statistic(rate, ts::AdaptiveTimeStep{false}) = maximum(rate)

function rate_statistic(rate, ts::AdaptiveTimeStep{true})
    copyto!(ts.host_rate, rate)
    n = length(ts.grounded_indices)
    for i in 1:n
        ts.scratch[i] = ts.host_rate[ts.grounded_indices[i]]
    end
    sort!(ts.scratch)
    idx = clamp(round(Int, ts.percentile * n), 1, n)
    return ts.scratch[idx]
end

"""
$(TYPEDSIGNATURES)

Solves for the new hydraulic head under [`EllipticHeadScheme`](@ref): runs the Picard/elliptic
solve (`sim.mi`'s melt input for the current time was already refreshed by [`step!`](@ref)).
"""
function step_h!(hs::EllipticHeadScheme, sim::Simulation)
    elliptic_solver!(hs.ps, sim.state, sim.grid, sim.p, sim.mt, sim.kfs, sim.sl; cnc = sim.cnc)
end

"""
$(TYPEDSIGNATURES)

Solves for the new hydraulic head under [`ParabolicHeadScheme`](@ref): a single backward-Euler
linear solve ([`parabolic_solver!`](@ref)), no Picard loop.
"""
function step_h!(hs::ParabolicHeadScheme, sim::Simulation)
    parabolic_solver!(hs.ls, sim.state, sim.grid, sim.p, sim.mt, sim.kfs, sim.sl, sim.dt[]; cnc = sim.cnc)
end

"""
$(TYPEDSIGNATURES)

Evolves the gap height `sim.state.b` by one timestep, dispatching on `sim.gs`
(`ImplicitGapScheme()`/`ExplicitGapScheme()`) to [`compute_b!(sim, sim.gs)`](@ref) below.

# Notes

Only evolves `b` where hydrology is actually being solved (`GROUNDED`). Cells with a
Dirichlet-prescribed `pw` (`LAND`/`OCEAN`) or a frozen `h` (`OTHER_BASIN`/`FROZEN_BED`) don't have a
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
    @parallel compute_b_implicit_kernel!(s.b, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt[], p.b_min, p.b_max)
    return sim
end

"""
$(TYPEDSIGNATURES)

Explicit (forward-Euler) update of `sim.state.b`: cheaper per step, but only stable for small enough `sim.dt`.
"""
function compute_b!(sim::Simulation, ::ExplicitGapScheme)
    s, p = sim.state, sim.p
    @parallel compute_b_explicit_kernel!(s.b, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt[], p.b_min, p.b_max)
    return sim
end

"""
$(TYPEDSIGNATURES)

Fully implicit update of `sim.state.b`: both the creep-closure term and the opening-by-sliding
term are evaluated at the new `b` -- unconditionally stable for any `sim.dt`, regardless of `p.br`.
Unlike [`ImplicitGapScheme`](@ref), this reads `p.br`/`p.lr` directly instead of `s.beta` (which
stays lagged by design, see `gap_height.jl`'s module docstring) -- see
[`compute_b_fully_implicit_kernel!`](@ref) for the closed-form two-branch solve.
"""
function compute_b!(sim::Simulation, ::FullyImplicitGapScheme)
    s, p = sim.state, sim.p
    @parallel compute_b_fully_implicit_kernel!(s.b, s.mask, s.mdot, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt[], p.b_min, p.b_max, p.br, p.lr)
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
    apply_cell_gap_clamping!(s, sim.cgc) # optional per-cell b_min/b_max override on top of p's global clamp -- no-op under the default NoCellGapClamping()

    compute_beta!(s, p, sim.oss) # opening-by-sliding parameter depends on the new b
    compute_b_x!(s)          # water depth on x faces
    compute_b_y!(s)          # water depth on y faces

end
