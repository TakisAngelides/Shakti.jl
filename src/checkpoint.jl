# Checkpoint/restart: periodically save everything needed to resume a
# Simulation's physics (full State, not just the observer's tracked_obs
# subset, since that's often a small slice picked for output/analysis, plus
# how far the time loop got) so a run killed at any tstep -- node failure,
# walltime limit, etc -- can pick back up instead of restarting from
# t=0. See run.jl for how tsteps/checkpoint_every/restart_path tie together,
# and observer.jl's resume!/reopenfile! for resuming the tracked-output file
# itself alongside the physics state.

"""
$(TYPEDSIGNATURES)

Saves everything needed to resume `sim`'s physics at timestep `t`: the full `State` (not just
the observer's `tracked_obs` subset, which is often a small slice picked for output/analysis),
`sim.total_time[]`, `sim.dt[]` (matters under [`AdaptiveTimeStep`](@ref): without it, a
resumed run would restart from the constructor's initial `dt` guess rather than whatever
`AdaptiveTimeStep` had actually converged to before the checkpoint), and the head-extrapolation
history ([`HeadExtrapolation`](@ref); without it a resumed run starts its next head solves from
another guess, and so ends them at another point within the tolerance).

# Notes

Written to a temp file and renamed into place (same filesystem, so `mv` is an atomic rename)
rather than written directly to `path`, so a crash mid checkpoint-write can't leave a
truncated/corrupt checkpoint behind -- the previous good checkpoint at `path` stays intact until
the new one has fully landed. See `run.jl` for how `tsteps`/`checkpoint_every`/`restart_path` tie
together, and `observer.jl`'s `resume!`/`reopenfile!` for resuming the tracked-output file itself
alongside the physics state.
"""
# The options that own State fields (cavity filling, uplift, gap coupling, face scheme, dry floor), saved with a checkpoint
# so a restart under other options says so: the next step zeroes the fields the new options do not own (prepare_step_inputs!).
option_signature(sim::Simulation) = join((nameof(typeof(sim.cf)), nameof(typeof(sim.up)), nameof(typeof(sim.gcp)),
                                          nameof(typeof(sim.kfs)), "dry_floor=$(dry_floor(sim))"), ", ")

function save_checkpoint(path::String, sim::Simulation, t::Int)
    state = sim.state
    tmp_path = path * ".tmp" # this works because jld2 recognizes its file not by the string extension but by a header inside the file, for other file types this might break
    JLD2.jldopen(tmp_path, "w") do file
        file["t"] = t
        file["total_time"] = sim.total_time[]
        file["dt"] = sim.dt[]
        file["options"] = option_signature(sim)
        save_head_history(file, sim.he)
        for name in fieldnames(typeof(state)) # for every field name in the State struct save it to file
            file[String(name)] = Array(getfield(state, name))
        end
    end
    mv(tmp_path, path; force = true) # we only update the file at path once we have finished writing to tmp_path so that there is no corruption possibility of the file at path
    return nothing
end

"""
$(TYPEDSIGNATURES)

Restores `sim.state`, `sim.total_time[]`, `sim.dt[]` and the head-extrapolation history in place from a checkpoint
written by [`save_checkpoint`](@ref), and returns the timestep it was saved at, so [`run!`](@ref) knows
where to resume the loop. A checkpoint written before `dt` was added to it (no `"dt"` key) leaves
`sim.dt[]` untouched, at whatever the `Simulation` constructor set it to; one without the history clears it
(`load_head_history!`). A checkpoint written under other options (cavity filling, uplift, gap coupling, face scheme, dry
floor) loads with a warning; the next step zeroes the fields the current options do not own.
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
        # bump (the bump height under the ice base) is lagged with beta, not refreshed by the next step's head refresh:
        # rebuild both from the loaded b and b_uplift when the checkpoint predates it
        haskey(file, "bump") || compute_beta!(state, sim.p, sim.oss)
        haskey(file, "mdot0") || copyto!(state.mdot0, state.mdot) # older checkpoints: the start-of-step melt of ImplicitUpwind is the saved melt
        haskey(file, "W") || sync_water!(state) # older checkpoints: the water W is the water the other fields hold
        load_head_history!(sim.he, file)
        if haskey(file, "options") && file["options"] != option_signature(sim)
            @warn "Checkpoint $path was written under other options; the fields the current ones do not own (b_empty, b_uplift, b_dry, stor, stor_up) are zeroed at the next step" saved = file["options"] current = option_signature(sim)
        end
        return file["t"], file["total_time"], (haskey(file, "dt") ? file["dt"] : nothing)
    end
    sim.total_time[] = total_time
    dt !== nothing && (sim.dt[] = dt)
    return t
end
