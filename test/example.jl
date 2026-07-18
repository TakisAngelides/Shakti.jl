import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Preferences
FT = Float64 # Float32, Float64
backend = "Threads" # "Threads", "Metal" - note Metal only works with Float32
set_preferences!("Shakti", "backend" => backend, "floattype" => string(FT); force=true) # only takes effect if julia is restarted i.e. not ran again from active REPL
println("Backend = $(backend) with $(FT), threads = $(Threads.nthreads())")

using Shakti
using CairoMakie

function main()

    # Simulation parameters
    nx = 16
    ny = 16
    lx = 1e3
    ly = 1e3
    tsteps = 24 * 24
    dt = 3600.0

    # Model parameters
    p = ModelParameters(e_v = 0.0) # e_v = 0.0 -> elliptic head scheme (Picard + linear solver, the only one implemented); e_v != 0.0 -> parabolic head scheme (not yet implemented)

    # Time integration scheme for the evolution of the water thickness (gap height) b
    gap_scheme_choice = "explicit" # "explicit" or "implicit"

    # Face-averaging scheme for the hydraulic conductivity K in the elliptic solve
    k_face_choice = "arithmetic" # "arithmetic" or "harmonic"

    # Grid and State setup
    grid = Grid(nx, ny, lx, ly)
    state = State(grid)

    # Melt input setup: reads directly off state.ieb (populated by the initial conditions below)
    mi = ConstantMeltInput()

    # =========================================================================
    # Initial conditions: sloped-slab synthetic glacier with one moulin at the
    # domain center, draining to an ocean margin at x = lx. A semicircular
    # "other basin" patch near x = 0 (rooted just outside the left boundary)
    # separates the modelled catchment from ice upstream of the moulin, and
    # the top/bottom domain edges are treated as inert side boundaries of that
    # catchment. Same synthetic test case as src_old/main.jl.
    # =========================================================================

    moulin_ij = (ceil(Int, nx / 2), ceil(Int, ny / 2))
    im, jm = moulin_ij
    xm, ym = grid.x[im], grid.y[jm]

    d  = 100.0   # leave 100 m of grounded ice upstream of the moulin
    xc = -200.0  # circle center outside the left boundary
    R  = xm - d - xc
    yc = ym

    mask = fill(GROUNDED, nx, ny)
    for j in 1:ny, i in 1:nx
        if (grid.x[i] - xc)^2 + (grid.y[j] - yc)^2 <= R^2
            mask[i, j] = OTHER_BASIN
        end
    end
    mask[end, :] .= OCEAN       # right edge: ocean margin, Dirichlet outlet
    mask[:, 1]   .= OTHER_BASIN # bottom edge: inert catchment boundary
    mask[:, end] .= OTHER_BASIN # top edge: inert catchment boundary

    slope         = 0.02
    ice_thickness = 500.0
    water_depth   = 0.01
    A_visc_val    = 5e-25

    A_visc = fill(A_visc_val, nx, ny)
    zb     = repeat(reshape(-slope .* grid.x, nx, 1), 1, ny) # bed slopes down toward the ocean margin
    zs     = zb .+ ice_thickness
    b      = fill(water_depth, nx, ny)
    G      = fill(0.06, nx, ny)
    ub_x   = fill(1e-6, nx + 1, ny)
    ub_y   = zeros(nx, ny + 1)
    ieb    = zeros(nx, ny)
    ieb[im, jm] = 3 / (grid.dx * grid.dy) # point moulin input, converted to a per-area rate

    set_initial_conditions!(state, grid, p, mi, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)

    # Linear solver
    ls = LUSolver(grid)

    # Picard solver
    iters = 100
    tol = 1e-6
    alpha = nothing # nothing -> no relaxation (NoHeadRelaxation); or a Float in (0,1) -> under-relaxation (UnderHeadRelaxation), damping h toward h_prev each Picard iteration
    ps = PicardSolver(iters, tol, ls; alpha = alpha)

    # Observer to track and plot state fields
    tracked_obs = ["h", "b"] # names of State fields to record; empty -> NoObserver
    which_observer = "Live" # "Live" (keep history in RAM) or "IO" (write to disk, needs which_file_writer + path below)
    # which_file_writer = "NetCDF" # only needed when which_observer == "IO": "NetCDF", "HDF5", "JLD2", or "CSV"
    # path = joinpath(@__DIR__, "Figures", "output.nc") # only needed when which_observer == "IO"
    tracked_times = 0:1:tsteps

    # Print per-time-step diagnostics (time taken, Picard convergence, iteration count)
    verbose = true

    # Create the simulation
    sim = Simulation(grid, state, tsteps, dt, p, gap_scheme_choice, tracked_obs, mi; ps = ps, which_observer = which_observer, tracked_times = tracked_times, k_face_choice = k_face_choice, verbose = verbose)

    # Run the simulation
    run!(sim)

    # =========================================================================
    # Animations + plots
    # =========================================================================

    dir = joinpath(@__DIR__, "Figures")
    mkpath(dir)

    moulin_locations = get_moulin_ij(state)
    make_mp4_2d(sim.observer, "h", moulin_locations; filename = joinpath(dir, "head.mp4"))
    make_mp4_2d(sim.observer, "b", moulin_locations; filename = joinpath(dir, "gap_height.mp4"))
    make_mp4_mid(sim.observer, "b", ceil(Int, ny / 2), moulin_locations; filename = joinpath(dir, "gap_height_midline.mp4"))

    CairoMakie.activate!()
    fig = Figure(size = (1200, 600))
    ax1 = Axis(fig[1, 1], title = "Gap Height (m)", xlabel = "x (m)", ylabel = "y (m)")
    hm_b = CairoMakie.heatmap!(ax1, sim.grid.x, sim.grid.y, Array(sim.state.b)) # Array(...) is used to convert GPU-resident arrays to host-accessible arrays for plotting if applicable (e.g. under the Metal backend)
    Colorbar(fig[1, 2], hm_b)
    ax2 = Axis(fig[1, 3], title = "Head (m)", xlabel = "x (m)", ylabel = "y (m)")
    hm_h = CairoMakie.heatmap!(ax2, sim.grid.x, sim.grid.y, Array(sim.state.h))
    Colorbar(fig[1, 4], hm_h)
    save(joinpath(dir, "final_state.png"), fig)
    display(fig)

    println("Finished.")

end

main()
