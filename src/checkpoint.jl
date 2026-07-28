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
the observer's `tracked_obs` subset, which is often a small slice picked for output/analysis) and
`sim.total_time[]`.

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
        for name in fieldnames(typeof(state)) # for every field name in the State struct save it to file
            file[String(name)] = Array(getfield(state, name))
        end
    end
    mv(tmp_path, path; force = true) # we only update the file at path once we have finished writing to tmp_path so that there is no corruption possibility of the file at path
    return nothing
end

"""
$(TYPEDSIGNATURES)

Restores `sim.state` and `sim.total_time[]` in place from a checkpoint written by
[`save_checkpoint`](@ref), and returns the timestep it was saved at, so [`run!`](@ref) knows
where to resume the loop.
"""
function load_checkpoint!(sim::Simulation, path::String)
    state = sim.state
    t, total_time = JLD2.jldopen(path, "r") do file
        for name in fieldnames(typeof(state))
            # copyto! (not .=): JLD2 always hands back a plain CPU Array, and
            # broadcasting that into a GPU-resident field (CuArray/MtlArray)
            # tries to compile a GPU kernel with the CPU Array as an argument,
            # which fails (a Matrix isn't a bitstype). copyto! is CUDA.jl/
            # Metal.jl's actual supported host->device transfer instead of a
            # kernel launch, and is exactly equivalent to `.=` for same-backend
            # (Threads/Array) fields.
            copyto!(getfield(state, name), file[String(name)])
        end
        return file["t"], file["total_time"]
    end
    sim.total_time[] = total_time
    return t
end
