# Checkpoint/restart: periodically save everything needed to resume a
# Simulation's physics (full State, not just the observer's tracked_obs
# subset, since that's often a small slice picked for output/analysis, plus
# how far the time loop got) so a run killed at any tstep -- node failure,
# walltime limit, etc -- can pick back up instead of restarting from
# t=0. See run.jl for how tsteps/checkpoint_every/restart_path tie together,
# and observer.jl's resume!/reopenfile! for resuming the tracked-output file
# itself alongside the physics state.

# The options that own State fields (cavity filling, uplift, gap coupling, face scheme, dry floor), saved with a checkpoint
# so a restart under other options says so: the next step zeroes the fields the new options do not own and brings the
# water W within the new sheet (prepare_step_inputs!).
option_signature(sim::Simulation) = join((nameof(typeof(sim.cf)), nameof(typeof(sim.up)), nameof(typeof(sim.gcp)),
                                          nameof(typeof(sim.kfs)), "dry_floor=$(dry_floor(sim))"), ", ")


"""
$(TYPEDSIGNATURES)

Saves everything needed to resume `sim`'s physics at timestep `t`: the full `State` (not just
the observer's `tracked_obs` subset, which is often a small slice picked for output/analysis),
`sim.total_time[]`, `sim.dt[]` (matters under [`AdaptiveTimeStep`](@ref): without it, a
resumed run would restart from the constructor's initial `dt` guess rather than whatever
`AdaptiveTimeStep` had actually converged to before the checkpoint) with the state of its controller, and the head-extrapolation
history ([`HeadExtrapolation`](@ref); without it a resumed run starts its next head solves from
another guess, and so ends them at another point within the tolerance), and the running totals of the
[`WaterBudget`](@ref) (`sim.wb`).

# Notes

Written to a temp file and renamed into place (same filesystem, so `mv` is an atomic rename)
rather than written directly to `path`, so a crash mid checkpoint-write can't leave a
truncated/corrupt checkpoint behind -- the previous good checkpoint at `path` stays intact until
the new one has fully landed. See `run.jl` for how `tsteps`/`checkpoint_every`/`restart_path` tie
together, and `observer.jl`'s `resume!`/`reopenfile!` for resuming the tracked-output file itself
alongside the physics state.
"""
function save_checkpoint(path::String, sim::Simulation, t::Int)
    state = sim.state
    tmp_path = path * ".tmp" # this works because jld2 recognizes its file not by the string extension but by a header inside the file, for other file types this might break
    JLD2.jldopen(tmp_path, "w") do file
        file["t"] = t
        file["total_time"] = sim.total_time[]
        file["dt"] = sim.dt[]
        file["options"] = option_signature(sim)
        save_head_history(file, sim.he)
        save_timestep_control(file, sim.ts)
        for name in WATER_BUDGET_TOTALS
            file["wb_" * String(name)] = getfield(sim.wb, name)[]
        end
        for name in fieldnames(typeof(state)) # for every field name in the State struct save it to file
            file[String(name)] = Array(getfield(state, name))
        end
    end
    mv(tmp_path, path; force = true) # we only update the file at path once we have finished writing to tmp_path so that there is no corruption possibility of the file at path
    return nothing
end

# The water of a checkpoint written before State.W (GAP-5): the gap less its empty part and the dry floor, plus the
# uplift layer, each missing field zero.
function checkpoint_water(file)
    b = file["b"]
    split(name) = haskey(file, name) ? file[name] : zero(b)
    return max.(b .- split("b_empty") .- split("b_dry"), zero(eltype(b))) .+ split("b_uplift")
end

# AdaptiveTimeStep's controller: the step it asks for and its count of fast steps, so a resumed run chooses the steps an
# uninterrupted one would, and its count of rejected steps, which continues like the water budget's totals. A checkpoint
# without them restarts the controller from the loaded dt.
save_timestep_control(file, ::AbstractTimeStepScheme) = nothing
save_timestep_control(file, ts::AdaptiveTimeStep) = (file["dt_ctrl"] = ts.dt_ctrl[]; file["fast_count"] = ts.fast_count[]; file["rejected"] = ts.rejected[]; nothing)
load_timestep_control!(::AbstractTimeStepScheme, file) = nothing
function load_timestep_control!(ts::AdaptiveTimeStep, file)
    ts.dt_ctrl[] = haskey(file, "dt_ctrl") ? file["dt_ctrl"] : NaN
    ts.fast_count[] = haskey(file, "fast_count") ? file["fast_count"] : 0
    ts.rejected[] = haskey(file, "rejected") ? file["rejected"] : 0
    return nothing
end


"""
$(TYPEDSIGNATURES)

Restores `sim.state`, `sim.total_time[]`, `sim.dt[]`, the head-extrapolation history and the water budget's totals in place from a checkpoint
written by [`save_checkpoint`](@ref), and returns the timestep it was saved at, so [`run!`](@ref) knows
where to resume the loop. A checkpoint written before `dt` was added to it (no `"dt"` key) leaves
`sim.dt[]` untouched, at whatever the `Simulation` constructor set it to; one without the history clears it
(`load_head_history!`). A checkpoint written under other options (cavity filling, uplift, gap coupling, face scheme, dry
floor) loads with a warning; the next step zeroes the fields the current options do not own and brings the water `W`
within the current sheet. One written before `W` (GAP-5) takes it from the fields that then split the water,
`max(b - b_empty - b_dry, 0) + b_uplift`.
"""
function load_checkpoint!(sim::Simulation, path::String)
    state = sim.state
    t, total_time, dt = JLD2.jldopen(path, "r") do file
        for name in fieldnames(typeof(state))
            # copyto! (not .=): JLD2 always hands back a plain CPU Array, and
            # broadcasting that into a GPU-resident field (CuArray/MtlArray)
            # tries to compile a GPU kernel with the CPU Array as an argument,
            # which fails (a Matrix isn't a bitstype). copyto! is CUDA.jl/
            # Metal.jl's actual supported host->device transfer instead of a
            # kernel launch, and is exactly equivalent to `.=` for same-backend
            # (Threads/Array) fields.
            # A checkpoint written before a field existed (e.g. K_x/K_y) simply lacks it; every such
            # field is derived, and is rebuilt from b/h by the refresh at the start of the next step.
            haskey(file, String(name)) && copyto!(getfield(state, name), file[String(name)])
        end
        haskey(file, "W") || copyto!(state.W, checkpoint_water(file)) # older checkpoints: the water the split fields held
        # bump (the bump height under the ice base) is lagged with beta, not refreshed by the next step's head refresh:
        # rebuild both from the loaded b, h and W when the checkpoint predates it
        haskey(file, "bump") || compute_beta!(state, sim.p, sim.oss)
        haskey(file, "mdot0") || copyto!(state.mdot0, state.mdot) # older checkpoints: the start-of-step melt of ImplicitUpwind is the saved melt
        load_head_history!(sim.he, file)
        load_timestep_control!(sim.ts, file)
        for name in WATER_BUDGET_TOTALS # an older checkpoint without them leaves the totals counting from the restart
            key = "wb_" * String(name)
            haskey(file, key) && (getfield(sim.wb, name)[] = file[key])
        end
        if haskey(file, "options") && file["options"] != option_signature(sim)
            @warn "Checkpoint $path was written under other options; the next step zeroes the fields the current ones do not own (stor, stor_up) and brings the water W within the current sheet" saved = file["options"] current = option_signature(sim)
        end
        return file["t"], file["total_time"], (haskey(file, "dt") ? file["dt"] : nothing)
    end
    sim.total_time[] = total_time
    dt !== nothing && (sim.dt[] = dt)
    return t
end
