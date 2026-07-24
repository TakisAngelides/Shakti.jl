# checkpoint_every/checkpoint_path: if both given, sim.state is saved to
# checkpoint_path (overwriting the previous checkpoint -- only the latest is
# kept) every checkpoint_every tsteps, so a run killed at any point can be
# resumed via restart_path below instead of starting over from t=0.
#
# restart_path: if given, resumes from a checkpoint written by a *previous*
# run! call instead of starting fresh at t=0 -- loads sim.state/total_time
# and continues the loop from the checkpointed tstep + 1 up to sim.tsteps
# (still the *original* total step count, not "how many more steps to do").
# The observer's output file is reopened (not truncated) via resume! so it's
# appended to across the restart; see observer.jl for how each file writer
# handles a tstep that the crashed run may have already written.
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

# Dispatched (rather than an isa check) so this stays correct if another
# AbstractHeadScheme is ever added: EllipticHeadScheme has a PicardSolver to
# report on, ParabolicHeadScheme (not yet implemented, see step_h! below)
# doesn't.
picard_status(hs::EllipticHeadScheme) = (hs.ps.converged, hs.ps.last_iter)
picard_status(hs::ParabolicHeadScheme) = (missing, missing)

function step!(sim::Simulation)

    step_h!(sim.hs, sim)
    step_b!(sim)

end

function step_h!(hs::EllipticHeadScheme, sim::Simulation)
    update_ieb!(sim.mi, sim.state, sim.total_time[]) # no-op for ConstantMeltInput; rescales state.ieb for e.g. SeasonalMeltInput
    elliptic_solver!(hs.ps, sim.state, sim.grid, sim.p, sim.shs, sim.kfs, sim.mi, sim.sl)
end

function step_h!(hs::ParabolicHeadScheme, sim::Simulation)
    error("Parabolic head scheme is not yet implemented.") # TODO
end

# compute_b! dispatches on sim.gs (ImplicitGapScheme/ExplicitGapScheme, see
# simulation.jl and compute_fields.jl) internally, so step_b! itself doesn't
# need to branch on the gap scheme.
function step_b!(sim::Simulation)

    s, p = sim.state, sim.p

    compute_b!(sim)       # updates b based on the new state variables (GROUNDED cells only)

    compute_beta!(s, p)   # opening-by-sliding parameter depends on the new b
    compute_b_x!(s)       # water depth on x faces
    compute_b_y!(s)       # water depth on y faces

end
