"""
$(TYPEDSIGNATURES)

Runs `sim` from `t=0` (or from a checkpoint, see `restart_path`/`extend_path`) up to `sim.tsteps`,
calling [`step!`](@ref) each iteration and recording output via `sim.observer`.

# Notes

`checkpoint_every`/`checkpoint_path`: if both given, `sim.state` is saved to `checkpoint_path`
(overwriting the previous checkpoint -- only the latest is kept) every `checkpoint_every` tsteps,
so a run killed at any point can be resumed via `restart_path` below instead of starting over from
`t=0`. The loop's actual last step is also always checkpointed on top of that periodic schedule
(whether or not it happens to be a `checkpoint_every` multiple), so a run that finishes cleanly
never leaves its newest checkpoint stale. The same checkpoint also feeds `extend_path`, for the
different "run finished, now go further" case -- see that keyword's own paragraph for how the two
differ.

`restart_path`: if given, resumes from a checkpoint written by a *previous* `run!` call instead of
starting fresh at `t=0` -- loads `sim.state`/`sim.total_time` and continues the loop from the
checkpointed tstep + 1 up to `sim.tsteps` (still the *original* total step count, not "how many
more steps to do"). The observer's output file is reopened (not truncated) via `resume!` so it's
appended to across the restart; see `observer.jl` for how each file writer handles a tstep that
the crashed run may have already written. Use this when a run was *killed* partway to its original
target and you want to finish reaching that same target in the same output file.

`extend_path`: the other way to start from a checkpoint, for the different case of a run that
*completed* its original target and you now want to push it further (a new, larger `sim.tsteps`
than the run that wrote the checkpoint had). Loads `sim.state`/`sim.total_time`/`sim.dt` from the
checkpoint via the same [`load_checkpoint!`](@ref) `restart_path` uses, but then starts a *fresh*
tracked-output file (via `prepare!`/`observe!`, the same path a cold start takes) instead of
reopening the old one -- the old file's own preallocated frame count was sized for its original,
now-too-small target, and reopening a *different* path (the natural result of `sim.tsteps`
changing, if your own script's output filename encodes it, as e.g. `pan_antarctica_simulation.jl`'s
does) would fail outright since that path doesn't exist yet. The loop itself then runs fresh from
`t=1` to `sim.tsteps`, so `sim.tsteps` here means "how many *more* steps to run," not the original
run's total -- unlike `restart_path`. The new file's frame 0 is stamped with the *loaded*
`total_time` (not 0), so day-labels stay physically correct (e.g. day 200 -> 300, not 0 -> 100)
even though it's a separate file from the original run's own output; a reader that wants one
continuous timeseries needs to know to read both files in sequence. Mutually exclusive with
`restart_path` -- pass at most one.

Under [`AdaptiveTimeStep`](@ref), the loop can also end *before* `sim.tsteps` if `total_time[]`
reaches `sim.ts.target_time` first (the intended way to stop a time-targeted adaptive run, since
the true step count needed to cover `target_time` isn't known in advance) -- `sim.tsteps` itself
should still be sized generously from `AdaptiveTimeStep`'s own `dt_min` (see its docstring) so the
loop is a real, finite upper bound either way, not just a number that happens to be big enough.
"""
function run!(sim::Simulation; checkpoint_every::Union{Nothing, Int} = nothing, checkpoint_path::Union{Nothing, String} = nothing, restart_path::Union{Nothing, String} = nothing, extend_path::Union{Nothing, String} = nothing)

    if (checkpoint_every === nothing) != (checkpoint_path === nothing)
        error("checkpoint_every and checkpoint_path must be given together")
    end

    if restart_path !== nothing && extend_path !== nothing
        error("restart_path and extend_path are mutually exclusive -- restart_path resumes a run " *
              "killed partway toward its ORIGINAL target (reopens/appends to the same tracked " *
              "file); extend_path continues an already-COMPLETED run toward a NEW, larger target " *
              "(starts a fresh tracked file, preserving total_time)")
    end

    if restart_path === nothing && extend_path === nothing
        sim.total_time[] = zero(sim.dt[]) # reset so the same Simulation can be run! more than once, e.g. chained runs sharing one state
        reset_head_history!(sim.he) # a fresh start's history must not include heads from a previous run!
        reset_timestep!(sim.ts)
        prepare!(sim.observer, sim.state)
        observe!(sim.observer, sim.state, 0, sim.total_time[])
        start_t = 0
    elseif restart_path !== nothing
        start_t = load_checkpoint!(sim, restart_path)
        resume!(sim.observer, sim.state, start_t)
    else # extend_path !== nothing
        load_checkpoint!(sim, extend_path) # sets sim.state/sim.total_time[]/sim.dt[]; return value (the checkpoint's own tstep) intentionally unused -- this file's own step indexing starts fresh at 0 below
        prepare!(sim.observer, sim.state)
        observe!(sim.observer, sim.state, 0, sim.total_time[]) # frame 0 of the NEW file is stamped with the LOADED total_time, not 0 -- see extend_path's own docstring paragraph above
        start_t = 0
    end

    last_t = start_t

    for t in (start_t + 1):sim.tsteps

        # AdaptiveTimeStep's target_time-based stop: sim.tsteps is still a real, finite upper
        # bound (sized from dt_min, see AdaptiveTimeStep's docstring), this is just the ordinary
        # "reached the target early" exit. target_time(::FixedTimeStep) is `nothing`, so this is a
        # no-op under the default scheme. Checked before the step, so a restart from a checkpoint
        # already at the target (a requeued job) takes no step with dt = 0 (a singular head matrix).
        (tt = target_time(sim.ts)) !== nothing && sim.total_time[] >= tt && break

        step_time = @elapsed step!(sim)

        sim.total_time[] += sim.dt[]

        observe!(sim.observer, sim.state, t, sim.total_time[])

        if checkpoint_every !== nothing && t % checkpoint_every == 0
            save_checkpoint(checkpoint_path, sim, t)
        end

        last_t = t

        if sim.verbose
            converged, last_iter = picard_status(sim.hs)
            println("$t / $(sim.tsteps) completed in $(round(step_time; digits = 4))s. head solve converged: $converged in $last_iter iterations.$(timestep_report(sim.ts, sim))")
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

    end

    # The periodic save above only fires on multiples of checkpoint_every, which won't generally
    # include the loop's actual last step (sim.tsteps itself, or an earlier AdaptiveTimeStep
    # target_time break) -- without this, a run that finishes cleanly can still leave its newest
    # checkpoint stale by up to checkpoint_every-1 steps, silently truncating any later extend_path
    # continuation to that older state instead of the true final one.
    if checkpoint_every !== nothing && last_t > start_t && last_t % checkpoint_every != 0
        save_checkpoint(checkpoint_path, sim, last_t)
    end

    finalize!(sim.observer, sim.state)

end

"""
$(TYPEDSIGNATURES)

Returns `(converged, last_iter)` for `hs`'s iterative solve at the current timestep -- dispatched
(rather than a field read) so this stays correct if another `AbstractHeadScheme` is ever added.
"""
picard_status(hs::PicardHeadScheme) = (hs.ps.converged, hs.ps.last_iter)

"""
$(TYPEDSIGNATURES)

Advances `sim` by one timestep: recomputes `sim.dt[]` if `sim.ts` is adaptive
(`update_dt!`), refreshes the melt input ([`update_ieb!`](@ref)), brings the head solve's
inputs up to date and sets its initial guess (`prepare_head_solve!`), solves for the new
head ([`step_h!`](@ref)), under [`AdaptiveTimeStep`](@ref) retrying it at half the `dt` while it does not converge
(`retry_step!`), records it for the next step's extrapolation (`sim.he`), evolves the gap height and the
water `W` ([`step_b!`](@ref)), checks the gap (`check_gap_height`) and lets an adaptive `dt` grow (`control_dt!`).
`sim.dt[]` is the step taken.
"""
function step!(sim::Simulation)
    compute_abs_ub!(sim.state) # a coupler may have set new ub_x/ub_y: opening by sliding, the freeze-on cap and update_dt! read |u_b|
    update_dt!(sim) # no-op under FixedTimeStep; must run before step_h!/step_b! since both read sim.dt[]
    update_ieb!(sim.mi, sim.state, sim.total_time[]) # writes Shakti's own input (state.ieb_own) for e.g. SeasonalMeltInput, then state.ieb = ieb_own + ieb_external -- done once per timestep, before step_h!, since ieb only feeds the head equation (step_b! never reads it)
    prepare_head_solve!(sim)
    step_h!(sim.hs, sim)
    while retry_step!(sim, sim.ts) # AdaptiveTimeStep: the head is back at the start of the step, dt halved
        prepare_head_solve!(sim)
        step_h!(sim.hs, sim)
    end
    store_start_of_step_melt!(sim.kfs, sim.state) # ImplicitUpwind: the converged melt, the next step's start-of-step melt (unfilled_cavities.jl); taken here so that a refresh_state! between steps cannot replace it
    record_head!(sim.he, sim.state, sim.dt[], first(picard_status(sim.hs)))
    step_b!(sim)
    check_gap_height(sim)
    control_dt!(sim, sim.ts)
    return sim
end

# The water W (GAP-5) is evolved from the budget with UnfilledCavities (update_water_budget!, less the change of the
# englacial water under e_v > 0).
# Elsewhere it is the water the head row stores: the gap scheme's water under the dry floor (compute_b_dry_floor!, set
# with the gap), otherwise the gap less the empty depth below the bed plus the uplift layer above the overburden head
# (water_from_head!). Under ConsistentGapCoupling that closes the budget to the Picard tolerance (1e-7 to
# 1e-6 of W on a moulin fixture) once the step converges, and stays bounded where it does not (the budget's W then holds
# the water the unconverged fluxes create: 0.8 m from a 0.03 m moulin input in one step of a stuck filled sheet); under
# ExplicitGapCoupling it does not close (2 % of W on that fixture).
water_is_budgeted(sim::Simulation) = sim.cf isa UnfilledCavities
function update_water!(sim::Simulation)
    water_is_budgeted(sim) && return update_water_budget!(sim.state, sim.grid, sim.p, sim.dt[], sim.hs.ps.h_old)
    dry_floor(sim) && return sim.state # compute_b! set it
    return water_from_head!(sim.state, sim.p, sim.cf, sim.up)
end

# A runaway gap: water above the overburden (N < 0) makes creep open the gap, and the implicit and fully implicit gap
# schemes grow it as exp(|C| dt) at the step's N, which the explicit coupling's head rows (linear growth) never see, so N
# does not recover. With b_max = Inf this reached 1e269 m in one 1-day step on a 1 m^3/s moulin slab (filled, NoUplift).
# The consistent coupling keeps that slab at metres, but at large enough dt its Picard iteration stops converging and the
# gap runs away as well (Helheim at 500 m, filled, 10-day steps), so the hint drops it once it is on.
gap_runaway_hint(::AbstractGapCoupling) = "ConsistentGapCoupling() (the head rows then take the gap the scheme reaches, and N recovers), " *
                                          "IceUplift() (water above the overburden lifts the ice instead), a finite b_max, or a smaller dt"
gap_runaway_hint(::ConsistentGapCoupling) = "a smaller dt (check that the head solve converged), IceUplift() (water above the overburden " *
                                            "lifts the ice instead), or a finite b_max"
@noinline _gap_not_finite(gcp) = error("step!: the gap height b is not finite in a GROUNDED cell. The usual cause is a gap runaway at N < 0: use " * gap_runaway_hint(gcp) * ".")
@noinline _warn_gap_thicker_than_ice(gcp) =
    @warn "step!: the gap height b exceeds the ice thickness in a GROUNDED cell, likely a gap runaway at N < 0; consider " * gap_runaway_hint(gcp) maxlog = 1

"""
$(TYPEDSIGNATURES)

Checks the gap after [`step_b!`](@ref): errors if `b` is not finite in a `GROUNDED` cell, and warns (once per session) if
it exceeds the ice thickness `H` there. Both point to a gap runaway at `N < 0` and its remedies. One lazy reduction over
the grid, no allocation.
"""
function check_gap_height(sim::Simulation)
    s = sim.state
    grounded = GROUNDED
    bad = Broadcast.instantiate(Broadcast.broadcasted((m, b, H) -> m == grounded ? (isfinite(b) ? (b > H ? 1 : 0) : 2) : 0, s.mask, s.b, s.H))
    worst = mapreduce(identity, max, bad; init = 0)
    worst == 2 && _gap_not_finite(sim.gcp)
    worst == 1 && _warn_gap_thicker_than_ice(sim.gcp)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Brings everything the head solve reads up to date with the state as the timestep starts -- `b`
has just been evolved by the previous [`step_b!`](@ref), and an ice-flow model may have changed
`zs`/`zb`/`u_b` since:

1. `H`/`po` from the current `b` ([`compute_H!`](@ref)/[`compute_po!`](@ref); `H = zs - (zb + b)`),
   so a large gap (a wide channel at fine resolution) lowers the overburden it sits under. Two
   elementwise kernels, negligible next to one linear solve. Under [`SheetTopPotential`](@ref) also
   the gap jumps across faces ([`update_gap_jumps!`](@ref)) from the same `b`, so that the flux is
   driven by `h + b` throughout the step.
2. The initial head guess ([`extrapolate_head!`](@ref), per `sim.he`, after the head at the start of the step is kept
   in the solver's `h_old`; a no-op under
   [`NoHeadExtrapolation`](@ref) or until two converged heads exist).
3. Every field derived from `h` and `b` ([`refresh_head_dependents!`](@ref): `pw`, `N`, face
   transmissivities/`q`/`Re`, `taub`, `mdot`, `K`, `D`). The first linear solve of the step
   assembles from these, so without this refresh it would use transmissivity, melt and `D`
   computed from the *previous* step's `b`. The solve recomputes them after every iteration anyway;
   this makes the first iteration consistent too (and hence the result of a solve that converges
   in one iteration). One refresh per step, a small fraction of a single iteration's cost.
"""
function prepare_head_solve!(sim::Simulation)
    s = sim.state
    prepare_step_inputs!(sim)
    copyto!(sim.hs.ps.h_old, s.h) # the head at the start of the step, for the englacial storage and the water test, before the guess replaces it
    extrapolate_head!(sim.he, s, sim.dt[])
    fill!(s.dh_it, 0) # the oscillation damping of the guess limiter must not see the previous step's last head step (14 % fewer iterations at 6 h steps)
    limit_head_guess!(picard_guard(sim), sim.he, s, sim.p) # the extrapolated head obeys the regime and outflow stops of the iteration
    refresh_from_head!(sim)
    return sim
end

# The step's inputs that depend on b and the mask only (step 1 of prepare_head_solve!).
function prepare_step_inputs!(sim::Simulation)
    s = sim.state
    compute_H!(s)
    compute_po!(s, sim.p)
    # The option-owned fields (stor, stor_up) are read by the kernels whatever the option, so each 'off' option writes its
    # neutral value, and the water W is brought within the sheet: a State that ran under another sheet (spin-up,
    # checkpoint) keeps no hybrid rows.
    set_uplift_storage!(s, sim.dt[], sim.up) # 1/dt marks cells where the ice may be lifted (zero under NoUplift)
    set_cavity_storage!(s, sim.dt[], sim.cf) # 1/dt marks cells that may be partly empty, read by the head rows and compute_pw!
    constrain_water!(s, sim.p, sim.cf, sim.up, dry_floor(sim)) # a no-op on the water the sheet's own steps leave
    update_gap_jumps!(s, sim.hp) # SheetTopPotential: gap jumps across faces from the start-of-step b, added to the driving gradient (no-op under BedPotential)
    if sim.p.limit_freeze_on   # freeze-on floor for this step, from the start-of-step gap
        freeze_on_capacity!(s.mdot_min, sim)
        s.mdot_min .*= -sim.p.rho_i
    end
    return sim
end

# Every field derived from the head, with the simulation's own options (step 3 of prepare_head_solve!).
function refresh_from_head!(sim::Simulation)
    reset_donors!(sim.kfs, sim.state) # ImplicitUpwind: every face starts with the higher-potential cell as donor (water_flux.jl)
    refresh_head_dependents!(sim.state, sim.grid, sim.p, sim.mt, sim.kfs, sim.sl; cnc = sim.cnc, ds = sim.ds, gc = gap_coupling_terms(sim))
    return sim
end

"""
$(TYPEDSIGNATURES)

Recomputes every field derived from the current `h`, `b` and mask with the simulation's own options: its face scheme,
[`AbstractCellNClamping`](@ref), outflow-only faces, cavity filling, uplift, diffusion and gap coupling. The setup and mask
helpers ([`set_initial_conditions!`](@ref), [`freeze_cells!`](@ref), [`thaw_cells!`](@ref), [`set_mask!`](@ref),
[`update_frozen_mask!`](@ref)) know only the `State` and `ModelParameters`, so they leave `N` without the cell clamps,
the outflow-only faces open and the face conductances arithmetic. Every step refreshes them before its first solve
(`prepare_head_solve!`), but what is read before that saw the helper's version: the observer's frame 0, step 1's
start-of-step melt ([`ImplicitUpwind`](@ref)) and freeze-on capacity, and a coupled model reading `N` or `mdot` after a
mask change. Called by `Simulation` at construction and by the `Simulation` methods of the mask helpers; `h` and `b` are
not changed, nor is the start-of-step melt of [`ImplicitUpwind`](@ref) (`State.mdot0`, the melt at the start-of-step gap,
kept from the setup or the last converged step): the refreshed melt is that of the end-of-step gap at the current head, a
first iterate rather than a start state, and taken as the start-of-step melt it made step 1 of an uplift moulin fixture at
dt = 1 day run to 500 iterations instead of 38.
"""
function refresh_state!(sim::Simulation)
    sim.total_time[] == 0 && store_start_of_step_melt!(sim.kfs, sim.state) # the first step's start-of-step melt: the setup's, at the start-of-step gap
    compute_abs_ub!(sim.state)
    prepare_step_inputs!(sim) # also brings the water W within this simulation's sheet (a setup, a mask change)
    refresh_from_head!(sim)
    return sim
end

freeze_cells!(sim::Simulation, freeze_mask::AbstractMatrix{Bool}) = (freeze_cells!(sim.state, sim.p, freeze_mask); refresh_state!(sim))
thaw_cells!(sim::Simulation, thaw_mask::AbstractMatrix{Bool}) = (thaw_cells!(sim.state, sim.p, thaw_mask); refresh_state!(sim))
set_mask!(sim::Simulation, new_mask::AbstractMatrix) = (r = set_mask!(sim.state, sim.p, new_mask); refresh_state!(sim); r)
update_frozen_mask!(sim::Simulation, T_prime_b::AbstractMatrix) = (r = update_frozen_mask!(sim.state, sim.p, T_prime_b); refresh_state!(sim); r)


"""
$(TYPEDSIGNATURES)

Applies [`limit_head_step!`](@ref) to the extrapolated initial head of a step ([`extrapolate_head!`](@ref)), measured
from the last converged head, without the filled cap; a no-op without head extrapolation or an active `guard`. A head
that changed regime in the last step moved by tens of metres (the range of a filled cell is the whole overburden head),
and extrapolating that change into the uplifted regime gives a blister tens of metres deep (seen on Drang Drung at
dt = 24 h: a cell that filled in step 7 started step 8 with an 84 m uplift layer, whose dissipation melt then fed the
layer faster than the iteration could drain it). The filled cap is left out: on the guess it doubled the iterations of
the dry floor on Drang Drung (25 m) and Helheim (200 m). The guess is a starting point only, so the solution is unchanged.
"""
limit_head_guess!(::PicardGuard, he, s::State, p::ModelParameters) = s
limit_head_guess!(guard::PicardGuard, he::HeadExtrapolation, s::State, p::ModelParameters) =
    he.count >= 1 ? limit_head_step!(guard, s, he.hist[1], p; filled_cap = false) : s

"""
$(TYPEDSIGNATURES)

The [`PicardGuard`](@ref) of `sim`: the limits of the head iterate that its sheet, gap coupling, face scheme and boundary
faces need ([`limit_head_step!`](@ref)). The filled cap is left out with englacial storage (`p.e_v > 0`) on SHAKTI's filled sheet
([`FilledCavities`](@ref), [`NoUplift`](@ref)): the englacial storage keeps the filled rows steep in `N`, and with the
cap the stuck-cells fixture of the tests took 251 iterations per step instead of 4 at `e_v = 1e-4` (same solution).
"""
function picard_guard(sim::Simulation)
    p = sim.p
    filled_cap = sim.gcp isa ConsistentGapCoupling && !(p.e_v > 0 && sim.cf isa FilledCavities && sim.up isa NoUplift)
    active = sim.cf isa UnfilledCavities || sim.up isa IceUplift || filled_cap || p.outflow_only_land || p.outflow_only_ocean || sim.kfs isa ImplicitUpwind
    return PicardGuard(active, filled_cap, p.outflow_only_land, p.outflow_only_ocean)
end

"""
$(TYPEDSIGNATURES)

Sets `sim.dt[]` for the step about to start under [`AdaptiveTimeStep`](@ref); a no-op under [`FixedTimeStep`](@ref)
(`dt` never changes). The step is the one the controller asks for (`ts.dt_ctrl[]`: the `Simulation`'s `dt` at the
first step, then halved by `retry_step!` and grown by `control_dt!`), capped, if `ts.safety_factor > 0`, by
`safety_factor/stat` (no lower than `dt_min`), where `stat` reduces the gap's relaxation rate `C+gamma` at the start of
the step -- the previous step's converged `N`/`u_b` -- ([`compute_dt_rate_kernel!`](@ref)) to one number
([`rate_statistic`](@ref)). The step is then fitted to `target_time` (`fit_to_target`).
"""
update_dt!(sim::Simulation) = update_dt!(sim, sim.ts)
update_dt!(sim::Simulation, ::FixedTimeStep) = sim

function update_dt!(sim::Simulation, ts::AdaptiveTimeStep)
    isnan(ts.dt_ctrl[]) && (ts.dt_ctrl[] = clamp(sim.dt[], ts.dt_min, ts.dt_max)) # the first step: the Simulation's dt
    dt = ts.dt_ctrl[]
    if ts.safety_factor > 0
        s, p = sim.state, sim.p
        @parallel compute_dt_rate_kernel!(ts.rate_field, s.mask, s.b, s.abs_ub, s.A_visc, s.N, p.n_minus_1_exp, s.bump, p.lr, sim.cls, p.b_c, p.b_min)
        stat = rate_statistic(ts.rate_field, ts)
        stat > 0 && (dt = min(dt, max(ts.safety_factor / stat, ts.dt_min))) # stat <= 0: no rated cell has a positive rate (all at b_min, or N <= 0), no cap
    end
    sim.dt[] = fit_to_target(ts, dt, ts.target_time - sim.total_time[])
    return sim
end

# The step dt fitted to the time rem left until target_time: the final step lands exactly on target_time instead of
# overshooting it. When the time left after this step would be less than dt_min, this step is shortened so the last one
# is exactly dt_min; with less than 2dt_min left it takes all of it (at most dt_max; beyond that, two halves). So no step
# falls below dt_min unless less than dt_min was left to begin with, dt_max < 2dt_min, or a rejected step was halved
# into the remainder.
function fit_to_target(ts::AdaptiveTimeStep, dt, rem)
    rem <= dt && return rem
    rem - dt >= ts.dt_min && return dt
    return rem - ts.dt_min >= ts.dt_min ? rem - ts.dt_min : # leave exactly dt_min for the last step
           rem <= ts.dt_max             ? rem :             # less than 2dt_min left: take it all now
                                          rem / 2           # ... unless that exceeds dt_max (only when dt_max < 2dt_min)
end

"""
$(TYPEDSIGNATURES)

Whether the step just solved for the head is rejected and retried. Under [`AdaptiveTimeStep`](@ref) a head solve that did
not converge, at a `dt` above `dt_min`, is: `dt` is halved (no lower than `dt_min`), the controller restarts its count of
fast steps from there, and the head is reset to the start of the step (`hs.ps.h_old`). Nothing else needs restoring: the
gap, the water `W`, the start-of-step melt and the head history change only once the head is accepted, and every field
derived from the head is refreshed from it by the next [`prepare_head_solve!`](@ref). Never under [`FixedTimeStep`](@ref).
"""
retry_step!(sim::Simulation, ::FixedTimeStep) = false
function retry_step!(sim::Simulation, ts::AdaptiveTimeStep)
    (first(picard_status(sim.hs)) || sim.dt[] <= ts.dt_min) && return false
    sim.dt[] = max(sim.dt[] / 2, ts.dt_min)
    ts.dt_ctrl[] = sim.dt[]
    ts.fast_count[] = 0
    ts.rejected[] += 1
    copyto!(sim.state.h, sim.hs.ps.h_old)
    return true
end

"""
$(TYPEDSIGNATURES)

After an accepted step under [`AdaptiveTimeStep`](@ref): counts the steps in a row whose head solve converged within
`ts.fast_iters` iterations, and after `ts.grow_after` of them grows the controller's step by `ts.grow` (at most `dt_max`).
A no-op under [`FixedTimeStep`](@ref).
"""
control_dt!(sim::Simulation, ::FixedTimeStep) = sim
function control_dt!(sim::Simulation, ts::AdaptiveTimeStep)
    converged, iters = picard_status(sim.hs)
    if converged && iters <= ts.fast_iters
        ts.fast_count[] += 1
        if ts.fast_count[] >= ts.grow_after
            ts.dt_ctrl[] = min(ts.dt_ctrl[] * ts.grow, ts.dt_max)
            ts.fast_count[] = 0
        end
    else
        ts.fast_count[] = 0
    end
    return sim
end

# A fresh run! starts the controller from the Simulation's dt again.
reset_timestep!(::FixedTimeStep) = nothing
reset_timestep!(ts::AdaptiveTimeStep) = (ts.dt_ctrl[] = NaN; ts.fast_count[] = 0; ts.rejected[] = 0; nothing)

# The timestep part of run!'s verbose line.
timestep_report(::FixedTimeStep, sim) = ""
timestep_report(ts::AdaptiveTimeStep, sim) = " dt = $(round(sim.dt[], sigdigits = 4)) s ($(ts.rejected[]) steps rejected so far)."

"""
$(TYPEDSIGNATURES)

Reduces `rate` (this step's `C+gamma` field, see [`compute_dt_rate_kernel!`](@ref)) to the single
statistic `update_dt!` divides `safety_factor` by. `UsePercentile=false`: the plain domain
maximum -- a single GPU-native reduction, no allocation, fully conservative (governed by the
single fastest-relaxing cell). `UsePercentile=true`: one bulk `copyto!` of `rate` into
`ts.host_rate` (CUDA.jl/Metal.jl's real device->host transfer -- scalar-indexing `rate` directly
per grounded cell would be illegal/catastrophically slow on a GPU backend), then a plain CPU loop
gathers the rated cells' values (`GROUNDED` with the gap above `b_min`: those [`compute_dt_rate_kernel!`](@ref) did
not mark `-Inf`, so the set follows mask changes during the run) into `ts.scratch`, sorts them in place,
and indexes the `ts.percentile`-th entry. None of this allocates -- every buffer is preallocated
once by [`AdaptiveTimeStep`](@ref)'s constructor. Returns `0` outright if there is no rated
cell, which `update_dt!` takes as "no cap", as it does "no cell has a positive rate" (the maximum of an all-`-Inf` field
is `-Inf`, so `UsePercentile=false` falls back the same way).
"""
rate_statistic(rate, ts::AdaptiveTimeStep{false}) = maximum(rate)

function rate_statistic(rate, ts::AdaptiveTimeStep{true})
    copyto!(ts.host_rate, rate)
    n = 0
    for r in ts.host_rate
        r == -Inf && continue # not grounded
        n += 1
        ts.scratch[n] = r
    end
    n == 0 && return zero(eltype(ts.scratch))
    grounded = view(ts.scratch, 1:n)
    sort!(grounded)
    idx = clamp(round(Int, ts.percentile * n), 1, n)
    return grounded[idx]
end

"""
$(TYPEDSIGNATURES)

Solves for the new hydraulic head: runs the Picard loop ([`elliptic_solver!`](@ref)) with the englacial storage over
`sim.dt[]` from the head at the start of the step that [`prepare_head_solve!`](@ref) stored in `hs.ps.h_old` (`sim.mi`'s
melt input for the current time was already refreshed by [`step!`](@ref)).
"""
function step_h!(hs::PicardHeadScheme, sim::Simulation)
    elliptic_solver!(hs.ps, sim.state, sim.grid, sim.p, sim.mt, sim.kfs, sim.sl; cnc = sim.cnc, ds = sim.ds, gc = gap_coupling_terms(sim), dt = sim.dt[], h_old_set = true,
                     wdt = water_check_dt(sim), guard = picard_guard(sim))
end

# The time step for the water test of the Picard loop (water_converged, elliptic_solver.jl), 0 where it does not apply: the
# head rows conserve exactly the conductance's water depth b_w only under ImplicitUpwind with ConsistentGapCoupling, and
# the water at the start of the step (W) is budgeted with these rows only under UnfilledCavities; a binding N clamp's held rows conserve
# nothing. Under IceUplift with UnfilledCavities N is in [0, po] already, so a floor at or below 0 and no cap never binds.
water_check_dt(sim::Simulation) = sim.kfs isa ImplicitUpwind && sim.gcp isa ConsistentGapCoupling && sim.cf isa UnfilledCavities &&
                                  N_clamps_inactive(sim) ? sim.dt[] : zero(sim.dt[])
function N_clamps_inactive(sim::Simulation)
    lo, hi = N_clamp_range(sim.cnc)
    lo, hi = max(lo, sim.p.N_min), min(hi, sim.p.N_max)
    return hi == Inf && (sim.up isa IceUplift ? lo <= 0 : lo == -Inf)
end

"""
$(TYPEDSIGNATURES)

Evolves the gap height `sim.state.b` by one timestep. Dispatches on `sim.ds` first: under
[`WithDiffusion`](@ref), `sim.gs` is **not consulted at all** -- [`solve_b_diffusion!`](@ref) solves
the coupled diffusion system (SUHMO Eq. 17) directly, since that equation only defines one way to
combine diffusion with the local opening/closure terms (closure with `N > 0` and
opening-by-sliding implicit on the diagonal, melt explicit -- see `b_diffusion_local_terms`,
`linear_solver.jl`), not one per
[`AbstractGapScheme`](@ref). Under [`NoDiffusion`](@ref) (the default), behaves exactly as before,
dispatching on `sim.gs` (`ImplicitGapScheme()`/`ExplicitGapScheme()`/`FullyImplicitGapScheme()`) to
[`compute_b!(sim, sim.gs)`](@ref) below.

# Notes

Only evolves `b` where hydrology is actually being solved (`GROUNDED`). Cells with a
Dirichlet-prescribed `pw` (`LAND`/`OCEAN`) or a frozen `h` (`OTHER_BASIN`/`FROZEN_BED`) don't have a
meaningfully-evolving `b` in this model, so their `b` is simply left untouched at whatever it was
initialized to (under [`WithDiffusion`](@ref): held fixed via its own frozen identity row instead,
same effect).
"""
compute_b!(sim::Simulation) = sim.ds isa WithDiffusion ? compute_b!(sim, sim.ds) : compute_b!(sim, sim.gs)

"""
$(TYPEDSIGNATURES)

Freeze-on capacity [m/s of ice] of every `GROUNDED` cell for a host's capacity basal boundary
condition, from the current state (call it before the host's thermodynamic step, i.e. with this
step's `b` and the last step's `N` and `|u_b|`): the fastest freezing the next gap update can absorb
without `b` falling below `p.b_min`, see `freeze_on_capacity_cell`. Zero off `GROUNDED`.
Under [`UnfilledCavities`](@ref) (`water_limited = true`, the default; `false` restores the gap-room limit alone) the capacity is also capped by the water actually present, `(rho_w/rho_i)*W/dt` (`s.W`): freezing
consumes water, which an empty cavity does not have. Without the cap a dry cell can freeze water that is not there (about 3 times the melt supply on Greenland 16 km).
Freezing closes the gap by the ice thickness and expels the excess water into the flow, so the limit
is gap room, not water mass, and the result is local (no inflow term). Under the dry floor (`dry_floor`) the water
`W` replaces `b - b_min` as the room, and the capacity is capped by the water too, `(rho_w/rho_i)*W/dt`
(`freeze_on_capacity_dry_floor_kernel!`): a gap the update may drain to zero can freeze only the water it holds.

Where each term is evaluated follows the gap scheme in use: [`FullyImplicitGapScheme`](@ref) takes
opening by sliding and creep at `b_min`, [`ImplicitGapScheme`](@ref) sliding at the old `b` (lagged
`beta`) and creep at `b_min`, [`ExplicitGapScheme`](@ref) both at the old `b`. Under [`WithDiffusion`](@ref) it follows SUHMO's
local terms (`freeze_on_capacity_cell_diffusion`) and ignores the diffusion of `b`, which only
moves gap volume between grounded cells: `C` is then conservative where `b` is a local minimum and
generous where it is a local maximum.

`dt` [s] is the interval the host applies `C` over, used for the gap-room term `(b - b_min)/dt`;
it defaults to Shakti's own step. When Shakti subcycles within a longer host step (e.g. Yelmo
advancing in `dt_min` chunks while Shakti steps hours), pass the host's step: the room in the gap
is a stock, available once per host step, so spreading it over Shakti's short step would overstate
`C` by the ratio of the two steps. With such a longer `dt`, opening by sliding and creep are also
evaluated at the current gap rather than at `b_min`: over many Shakti steps creep keeps closing the
gap at its actual size, and evaluating it at `b_min` (right for one implicit step) overstated `C`.
"""
function freeze_on_capacity!(C, sim::Simulation; dt = sim.dt[], water_limited = true)
    s, p = sim.state, sim.p
    if dry_floor(sim) # the water W may drain below b_min: freeze no more than W leaves room for, nor more water than W
        @parallel freeze_on_capacity_dry_floor_kernel!(C, s.mask, s.W, s.beta, s.abs_ub, s.A_visc, s.N, p.n_minus_1_exp, dt,
                                                       sim.cls, p.b_c, sim.gs isa ExplicitGapScheme, p.rho_w / p.rho_i)
        return C
    end
    diffusion = sim.ds isa WithDiffusion
    # Over a host step longer than Shakti's own, creep and sliding act on the current gap for most
    # of the interval, not on b_min (the end state of a single implicit step), so evaluate them at b.
    subcycled = dt > sim.dt[]
    beta_at_bmin, creep_at_bmin = subcycled ? (false, false) :
                                  sim.gs isa FullyImplicitGapScheme ? (true, true) :
                                  sim.gs isa ImplicitGapScheme ? (false, true) : (false, false)
    @parallel freeze_on_capacity_kernel!(C, s.mask, s.b, s.beta, s.lc, s.abs_ub, s.A_visc, s.N, p.n_minus_1_exp, dt,
                                         p.b_min, s.bump, p.lr, sim.cls, p.b_c, beta_at_bmin, creep_at_bmin, diffusion,
                                         s.W, p.rho_w / p.rho_i, sim.cf isa UnfilledCavities && water_limited)
    return C
end

"""
$(TYPEDSIGNATURES)

Solves `b`'s coupled diffusion system for the new timestep ([`solve_b_diffusion!`](@ref), using
the second, independent linear solver bundled in `ds`) -- the [`WithDiffusion`](@ref) counterpart
of the `sim.gs`-dispatched methods below, taking over `b`'s evolution entirely rather than
composing with any of them.
"""
function compute_b!(sim::Simulation, ds::WithDiffusion)
    solve_b_diffusion!(ds.ls, sim.state, sim.grid, sim.p, sim.dt[])
    return sim
end

"""
$(TYPEDSIGNATURES)

Implicit (backward-Euler) update of `sim.state.b`: implicit on the creep closure term only. Deprecated, see
[`ImplicitGapScheme`](@ref); [`FullyImplicitGapScheme`](@ref) is the default.
"""
function compute_b!(sim::Simulation, ::ImplicitGapScheme)
    dry_floor(sim) && return compute_b_dry_floor!(sim)
    s, p = sim.state, sim.p
    @parallel compute_b_implicit_kernel!(s.b, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt[], p.b_min, p.b_max, sim.cls, p.b_c)
    return sim
end

"""
$(TYPEDSIGNATURES)

Explicit (forward-Euler) update of `sim.state.b`: cheaper per step, but only stable for small enough `sim.dt`.
"""
function compute_b!(sim::Simulation, ::ExplicitGapScheme)
    dry_floor(sim) && return compute_b_dry_floor!(sim)
    s, p = sim.state, sim.p
    @parallel compute_b_explicit_kernel!(s.b, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt[], p.b_min, p.b_max, sim.cls, p.b_c)
    return sim
end

"""
$(TYPEDSIGNATURES)

Fully implicit update of `sim.state.b`: both the creep-closure term and the opening-by-sliding
term are evaluated at the new `b` -- unconditionally stable for any `sim.dt`, regardless of `p.br`.
Unlike [`ImplicitGapScheme`](@ref), this reads `p.br`/`p.lr` directly instead of `s.beta` (which
stays lagged by design, see `gap_height.jl`'s module docstring) -- see
`compute_b_fully_implicit_kernel!` for the closed-form solve (two branches under
[`StandardCreep`](@ref), four under [`CreepCutoff`](@ref)).
"""
function compute_b!(sim::Simulation, ::FullyImplicitGapScheme)
    dry_floor(sim) && return compute_b_dry_floor!(sim)
    s, p = sim.state, sim.p
    @parallel compute_b_fully_implicit_kernel!(s.b, s.mask, s.mdot, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt[], p.b_min, p.b_max, s.bump, p.lr, sim.cls, p.b_c)
    return sim
end

"""
$(TYPEDSIGNATURES)

One cell's gap after the step under the gap scheme `GS` of a [`SchemeGapUpdate`](@ref), before the `b_min`/`b_max` clamp:
the same formulas as [`compute_b!`](@ref)'s kernels above (`C = A*|N|^(n-1)*N`), so that the head rows of
[`ConsistentGapCoupling`](@ref) assume exactly the gap change the gap scheme then makes.
"""
@inline gap_update(gu::SchemeGapUpdate{ExplicitGapScheme, CLS}, b, mdot, beta, abs_ub, C, rho_i) where CLS =
    b + gu.dt * (mdot / rho_i + beta * abs_ub - C * creep_length(CLS(), b, gu.b_c, C))
@inline gap_update(gu::SchemeGapUpdate{ImplicitGapScheme, CLS}, b, mdot, beta, abs_ub, C, rho_i) where CLS =
    implicit_creep_update(CLS(), b, mdot / rho_i + beta * abs_ub, C, gu.dt, gu.b_c)
@inline gap_update(gu::SchemeGapUpdate{FullyImplicitGapScheme, CLS}, b, mdot, beta, abs_ub, C, rho_i) where CLS =
    fully_implicit_creep_update(CLS(), b, mdot / rho_i, abs_ub / gu.lr, C, gu.dt, gu.br, gu.b_c)

"""
$(TYPEDSIGNATURES)

The gap coupling the head rows of this step use: [`ExplicitGapCoupling`](@ref) as it is, or, for
[`ConsistentGapCoupling`](@ref), its kernel form [`SchemeGapUpdate`](@ref) with this step's `dt` and the simulation's gap
and creep-length schemes, or `DiffusiveGapUpdate` under [`WithDiffusion`](@ref).
"""
gap_coupling_terms(sim::Simulation) = gap_coupling_terms(sim, sim.gcp)
gap_coupling_terms(sim::Simulation, ::ExplicitGapCoupling) = ExplicitGapCoupling()
gap_coupling_terms(sim::Simulation, gcp::ConsistentGapCoupling) = gap_coupling_terms(sim, gcp, sim.ds)
gap_coupling_terms(sim::Simulation, ::ConsistentGapCoupling, ::NoDiffusion) =
    SchemeGapUpdate{typeof(sim.gs), typeof(sim.cls), typeof(sim.dt[])}(sim.dt[], sim.p.br, sim.p.lr, sim.p.b_c, dry_floor(sim))
gap_coupling_terms(sim::Simulation, ::ConsistentGapCoupling, ::WithDiffusion) = DiffusiveGapUpdate(sim.dt[], oftype(sim.dt[], sim.p.br), oftype(sim.dt[], sim.p.lr))

"""
$(TYPEDSIGNATURES)

Whether the `b_min` floor is dry: [`ConsistentGapCoupling`](@ref) under [`FilledCavities`](@ref) and [`NoUplift`](@ref).
The head rows then store the water `W` (`State.W`), moved by the gap scheme's update with no lower clamp
([`scheme_water`](@ref)), and [`compute_b!`](@ref) holds the gap at `b_min` with the volume below the water dry
(`b - W`), so the head equation and the gap update store the same water also where creep closes the gap within a
step. The conductances take the water too ([`compute_b_w!`](@ref), `end_of_step_gap`): a dry floor does not conduct.
Conducting at `K(b_min)` it relayed its neighbours' water at an almost flat head while its row had lost its storage
slope, which pinned Drang Drung at `b_min` with `p_w ~ -3 po` (b_min = 1e-3); with the conductance from the water, the
results there do not depend on `b_min`. Under [`UnfilledCavities`](@ref) the floor's volume
is part of the cavity's empty volume instead, and with [`IceUplift`](@ref) the floor stays full. Not under [`WithDiffusion`](@ref),
whose gap solve clamps at `b_min` ([`solve_b_diffusion!`](@ref)).
"""
dry_floor(sim::Simulation) = sim.gcp isa ConsistentGapCoupling && sim.cf isa FilledCavities && sim.up isa NoUplift && sim.ds isa NoDiffusion

@parallel_indices (ix, iy) function compute_b_dry_floor_kernel!(b, W, mask, mdot, beta, abs_ub, A_visc, N, rho_i, n_minus_1, b_min, b_max, gu, bump)
    if ix <= size(b, 1) && iy <= size(b, 2)
        if mask[ix, iy] == GROUNDED
            Wn = scheme_water(cell_gap(gu, bump[ix, iy]), b[ix, iy], W[ix, iy], b_min, b_max, mdot[ix, iy], beta[ix, iy], abs_ub[ix, iy], A_visc[ix, iy], N[ix, iy], rho_i, n_minus_1)
            b[ix, iy] = max(Wn, b_min)
            W[ix, iy] = Wn
        else
            W[ix, iy] = b[ix, iy]
        end
    end
    return
end

"""
$(TYPEDSIGNATURES)

[`compute_b!`](@ref) under the dry floor (`dry_floor`), for every gap scheme: the water moves by the scheme's own
update with no lower clamp ([`scheme_water`](@ref) of the water `s.W`, the same function the head rows use) and is
written to `s.W`, and the gap is that water held at `b_min`, dry below the water. Off `GROUNDED` the water is the gap.
"""
function compute_b_dry_floor!(sim::Simulation)
    s, p = sim.state, sim.p
    @parallel compute_b_dry_floor_kernel!(s.b, s.W, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, p.b_min, p.b_max, gap_coupling_terms(sim), s.bump)
    return sim
end

"""
$(TYPEDSIGNATURES)

Evolves the gap height `sim.state.b` for one timestep ([`compute_b!`](@ref), dispatching
internally on `sim.ds`/`sim.gs`), sets the water `W` for it (`update_water!`), then refreshes everything that depends on it (`beta`, `lc`) so it is ready for the
*next* timestep's Picard loop, and the face diagnostics `b_x`/`b_y` (no longer read by the solve; the face conductances
take the water depth `b_w`, [`compute_face_flux!`](@ref)).
"""
function step_b!(sim::Simulation)

    s, p = sim.state, sim.p

    compute_b!(sim)          # updates b based on the new state variables (GROUNDED cells only)
    update_water!(sim)       # the water W for the new gap, before compute_beta! reads it

    compute_beta!(s, p, sim.oss) # opening-by-sliding parameter depends on the new b
    compute_lc!(s, p, sim.cls)   # ice-creep length scale depends on the new b
    compute_b_x!(s)          # water depth on x faces
    compute_b_y!(s)          # water depth on y faces

end
