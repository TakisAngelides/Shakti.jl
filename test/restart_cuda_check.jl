# Verifies checkpoint/restart works under the CUDA backend, not just Threads
# (which is all test/runtests.jl exercises, since it hardcodes the backend
# preference). Only CGIterativeSolver(g, MatrixFreeLinearSystem) is GPU-
# capable (CholeskyDirectSolver/CG+SALS are Threads/CPU-only by design), so
# that's the solver under test here.
#
# Mirrors test/runtests.jl's "Checkpoint / Restart" testset: an uninterrupted
# baseline run vs. a deliberately-crashed-then-resumed one (crash point NOT
# aligned to the checkpoint, forcing replay of already-observed tsteps),
# checking state, total_time, and the full NetCDF output file all match.

using Preferences
set_preferences!("Shakti", "backend" => "CUDA", "floattype" => "Float64"; force = true)

using Shakti
using NetCDF
using Printf

println("Active backend: ", Shakti.backend)

nx, ny = 6, 6
grid = Grid(nx, ny, 1e3, 1e3)
p = ModelParameters(e_v = 0.0)
mi = ConstantMeltInput()
sl = RegularizedCoulombSlidingLaw(0.25)

mask = fill(GROUNDED, nx, ny)
mask[end, :] .= OCEAN
mask[1, :]   .= LAND
mask[:, 1]   .= OTHER_BASIN

A_visc = fill(5e-25, nx, ny)
zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
zs     = zb .+ 500.0
b      = fill(0.01, nx, ny)
G      = fill(0.06, nx, ny)
ub_x   = fill(1e-6, nx + 1, ny)
ub_y   = zeros(nx, ny + 1)
ieb    = zeros(nx, ny)
ieb[3, 3] = 3 / (grid.dx * grid.dy)
taub_x = zeros(nx + 1, ny)
taub_y = zeros(nx, ny + 1)

dt = 3600.0
tsteps = 8
tracked_obs = ["h", "b"]
tracked_times = 0:tsteps

function make_sim(sim_tsteps, path)
    state = State(grid)
    set_initial_conditions!(state, grid, p, mi, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
    println("  state.h array type: ", typeof(state.h)) # confirms this is actually GPU-resident, not silently falling back to CPU
    ls = CGIterativeSolver(grid, MatrixFreeLinearSystem)
    ps = PicardSolver(500, 1e-6, ls, grid)
    return Simulation(grid, state, sim_tsteps, floattype(dt), p, "implicit", tracked_obs, mi, sl;
                       ps = ps, which_observer = "IO", which_file_writer = "NetCDF",
                       tracked_times = tracked_times, path = path)
end

mktempdir() do dir

    println("Building baseline (uninterrupted) simulation...")
    full_path = joinpath(dir, "full.nc")
    sim_full = make_sim(tsteps, full_path)
    run!(sim_full)
    println("Baseline done.")

    println("Building 'crashed' simulation (stops at t=5, checkpoints every 3)...")
    crash_path = joinpath(dir, "crash.nc")
    checkpoint_path = joinpath(dir, "ckpt.jld2")
    sim_crash = make_sim(5, crash_path)
    run!(sim_crash; checkpoint_every = 3, checkpoint_path = checkpoint_path)
    println("Crash simulated.")

    println("Resuming into a brand-new Simulation from the t=3 checkpoint...")
    sim_resumed = make_sim(tsteps, crash_path)
    run!(sim_resumed; restart_path = checkpoint_path)
    println("Resume done.")

    h_full = Array(sim_full.state.h)
    h_resumed = Array(sim_resumed.state.h)
    b_full = Array(sim_full.state.b)
    b_resumed = Array(sim_resumed.state.b)

    state_ok = isapprox(h_resumed, h_full) && isapprox(b_resumed, b_full)
    time_ok = isapprox(sim_resumed.total_time[], sim_full.total_time[])

    h_nc_full = NetCDF.ncread(full_path, "h")
    h_nc_resumed = NetCDF.ncread(crash_path, "h")
    file_ok = size(h_nc_resumed) == size(h_nc_full) && isapprox(h_nc_resumed, h_nc_full)

    @printf("\nstate.h/b match: %s\nstate.total_time match: %s\noutput file match (no duplicated/dropped tsteps): %s\n", state_ok, time_ok, file_ok)
    println(state_ok && time_ok && file_ok ? "\nALL CHECKS PASSED on CUDA backend" : "\nFAILURE on CUDA backend")

end
