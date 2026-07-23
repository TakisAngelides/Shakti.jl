function run!(sim::Simulation)

    sim.total_time[] = zero(sim.dt) # reset so the same Simulation can be run! more than once, e.g. chained runs sharing one state

    prepare!(sim.observer, sim.state)
    observe!(sim.observer, sim.state, 0, sim.total_time[])

    for t in 1:sim.tsteps

        step_time = @elapsed step!(sim)

        sim.total_time[] += sim.dt

        observe!(sim.observer, sim.state, t, sim.total_time[])

        if sim.verbose
            converged, last_iter = picard_status(sim.hs)
            println("$t / $(sim.tsteps) completed in $(round(step_time; digits = 4))s. Picard converged: $converged in $last_iter iterations")
            if converged === false
                s = sim.state
                println("  diagnostics: N=$(extrema(Array(s.N))) Re=$(extrema(Array(s.Re))) b=$(extrema(Array(s.b))) h=$(extrema(Array(s.h)))")
            end
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
