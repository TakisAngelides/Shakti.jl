function run!(sim::Simulation)

    F = typeof(sim.dt)
    total_time = F(0.0)

    prepare!(sim.observer, sim.state)
    observe!(sim.observer, sim.state, 0, total_time)

    for t in 1:sim.tsteps

        step_time = @elapsed step!(sim)

        total_time += sim.dt

        observe!(sim.observer, sim.state, t, total_time)

        if sim.verbose
            converged, last_iter = picard_status(sim.hs)
            println("$t / $(sim.tsteps) completed in $(round(step_time; digits = 4))s. Picard converged: $converged in $last_iter iterations")
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

    @timeit PERF_TIMER "step_h!" step_h!(sim.hs, sim)
    @timeit PERF_TIMER "step_b!" step_b!(sim)

end

function step_h!(hs::EllipticHeadScheme, sim::Simulation)
    elliptic_solver!(hs.ps, sim.state, sim.grid, sim.p, sim.shs, sim.kfs, sim.mi)
end

function step_h!(hs::ParabolicHeadScheme, sim::Simulation)
    error("Parabolic head scheme is not yet implemented.") # TODO
end

# compute_b! dispatches on sim.gs (ImplicitGapScheme/ExplicitGapScheme, see
# simulation.jl and compute_fields.jl) internally, so step_b! itself doesn't
# need to branch on the gap scheme.
function step_b!(sim::Simulation)

    s, p = sim.state, sim.p

    @timeit PERF_TIMER "compute_b!" compute_b!(sim)       # updates b based on the new state variables (GROUNDED cells only)

    @timeit PERF_TIMER "compute_beta!" compute_beta!(s, p)   # opening-by-sliding parameter depends on the new b
    @timeit PERF_TIMER "compute_b_x!" compute_b_x!(s)       # water depth on x faces
    @timeit PERF_TIMER "compute_b_y!" compute_b_y!(s)       # water depth on y faces

end
