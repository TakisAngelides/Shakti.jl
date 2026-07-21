import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Preferences

# ============================================================================
# PARAMETER OPTIONS - Modify columns below to test different configurations
# ============================================================================
# Each row is a complete configuration. Uncomment/comment or edit rows to
# customize. NOTE: backend and floattype are Julia preferences baked into
# Shakti at load time, so only ONE config can run per script invocation
# (set ACTIVE_LABEL below to pick which). To compare two backends/floattypes,
# run this script twice (e.g. two terminal invocations).

const CONFIGS = Dict(
    # label            => (FT,       backend,    nx,  ny,  gap_scheme,  K_face,       e_v,  alpha,  linear_solver,  mask_choice)
    "baseline_LU"       => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "LU",          "barrier"),
    "gmres_sparse"      => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "GMRES",       "barrier"),
    "gmres_mf"          => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "GMRES_MF",    "barrier"),
    "bicgstab_sparse"   => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "BiCGSTAB",    "barrier"),
    "bicgstab_mf"       => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "BiCGSTAB_MF", "barrier"),
    "simple_mask"       => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "LU",          "simple"),
    "harmonic_K"        => (Float64, "Threads",   32,  32,  "explicit",  "harmonic",   0.0,  nothing, "LU",          "barrier"),
    "implicit_gap"      => (Float64, "Threads",   32,  32,  "implicit",  "arithmetic", 0.0,  nothing, "LU",          "barrier"),
    "relaxation"        => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  0.5,     "LU",          "barrier"),
)

# ----------------------------------------------------------------------------
# Pick which config to run
# ----------------------------------------------------------------------------
const ACTIVE_LABEL = "gmres_sparse"

haskey(CONFIGS, ACTIVE_LABEL) || error("Unknown config label '$ACTIVE_LABEL'. Choose one of: $(join(keys(CONFIGS), ", "))")
const (FT, BACKEND, NX, NY, GAP_SCHEME, K_FACE, E_V, ALPHA, LINEAR_SOLVER, MASK_CHOICE) = CONFIGS[ACTIVE_LABEL]

# Preferences must be set before `using Shakti` to take effect
set_preferences!("Shakti", "backend" => BACKEND, "floattype" => string(FT); force = true)

using Shakti
using CairoMakie

# ============================================================================
# SIMULATION SETTINGS (fixed across configs)
# ============================================================================
const SIM_PARAMS = (
    lx = 1e3,
    ly = 1e3,
    tsteps = 24 * 24,
    dt = 3600.0,
    slope = 0.02,
    ice_thickness = 500.0,
    water_depth = 0.01,
    A_visc_val = 5e-25,
    picard_iters = 100,
    picard_tol = 1e-6,
    verbose = true,
    save_figs = true,
)

# ============================================================================
# MASK CONSTRUCTION
# ============================================================================
function build_mask(mask_choice, grid, nx, ny, im, jm)

    mask = fill(GROUNDED, nx, ny)

    if mask_choice == "simple"
        # Plain rectangle: ocean outlet on right, inert boundaries elsewhere
        mask[end, :] .= OCEAN
        mask[1, :]   .= OTHER_BASIN
        mask[:, 1]   .= OTHER_BASIN
        mask[:, end] .= OTHER_BASIN

    elseif mask_choice == "barrier"
        # Downstream obstacle that forces flow to split around it
        mask[20, 15] = OTHER_BASIN
        mask[20, 16] = OTHER_BASIN
        mask[20, 17] = OTHER_BASIN
        mask[1, :]   .= OTHER_BASIN
        mask[end, :] .= OCEAN
        mask[:, 1]   .= OTHER_BASIN
        mask[:, end] .= OTHER_BASIN

    elseif mask_choice == "semi-circle"
        # Semicircular upstream basin boundary near the left edge
        xm, ym = grid.x[im], grid.y[jm]
        d  = 100.0
        xc = -200.0
        R  = xm - d - xc
        yc = ym
        for j in 1:ny, i in 1:nx
            if (grid.x[i] - xc)^2 + (grid.y[j] - yc)^2 <= R^2
                mask[i, j] = OTHER_BASIN
            end
        end
        mask[end, :] .= OCEAN
        mask[:, 1]   .= OTHER_BASIN
        mask[:, end] .= OTHER_BASIN

    else
        error("Unknown mask_choice: $mask_choice. Choose from: \"simple\", \"barrier\", \"semi-circle\"")
    end

    return mask
end

# ============================================================================
# LINEAR SOLVER CONSTRUCTION
# ============================================================================
function build_linear_solver(linear_solver, grid)

    if linear_solver == "LU"
        LUDirectSolver(grid)
    elseif linear_solver == "GMRES"
        GMRESIterativeSolver(grid, SparseAssembledLinearSystem)
    elseif linear_solver == "GMRES_MF"
        GMRESIterativeSolver(grid, MatrixFreeLinearSystem)
    elseif linear_solver == "BiCGSTAB"
        BiCGSTABIterativeSolver(grid, SparseAssembledLinearSystem)
    elseif linear_solver == "BiCGSTAB_MF"
        BiCGSTABIterativeSolver(grid, MatrixFreeLinearSystem)
    else
        error("Unknown linear_solver: $linear_solver. Choose from: \"LU\", \"GMRES\", \"GMRES_MF\", \"BiCGSTAB\", \"BiCGSTAB_MF\"")
    end
end

# ============================================================================
# MAIN
# ============================================================================
function main()
    println("="^80)
    println("Running config: $ACTIVE_LABEL")
    println("Backend: $BACKEND ($FT), Grid: $(NX)×$(NY), Gap: $GAP_SCHEME, K_face: $K_FACE,")
    println("e_v: $E_V, alpha: $ALPHA, Solver: $LINEAR_SOLVER, Mask: $MASK_CHOICE")
    println("="^80)

    # Grid and state
    grid = Grid(NX, NY, SIM_PARAMS.lx, SIM_PARAMS.ly)
    state = State(grid)

    # Model parameters
    p = ModelParameters(e_v = E_V)

    # Melt input
    mi = ConstantMeltInput()

    # Mask
    moulin_ij = (ceil(Int, NX / 2), ceil(Int, NY / 2))
    im, jm = moulin_ij
    mask = build_mask(MASK_CHOICE, grid, NX, NY, im, jm)

    # Initial conditions
    A_visc = fill(SIM_PARAMS.A_visc_val, NX, NY)
    zb     = repeat(reshape(-SIM_PARAMS.slope .* grid.x, NX, 1), 1, NY)
    zs     = zb .+ SIM_PARAMS.ice_thickness
    b      = fill(SIM_PARAMS.water_depth, NX, NY)
    G      = fill(0.06, NX, NY)
    ub_x   = fill(1e-6, NX + 1, NY)
    ub_y   = zeros(NX, NY + 1)
    ieb    = zeros(NX, NY)
    ieb[im, jm] = 3 / (grid.dx * grid.dy)

    set_initial_conditions!(state, grid, p, mi, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)

    # Solvers
    ls = build_linear_solver(LINEAR_SOLVER, grid)
    ps = PicardSolver(SIM_PARAMS.picard_iters, SIM_PARAMS.picard_tol, ls, grid; alpha = ALPHA)

    # Observer and simulation
    tracked_obs = ["h", "b"]
    tracked_times = 0:1:SIM_PARAMS.tsteps

    sim = Simulation(grid, state, SIM_PARAMS.tsteps, SIM_PARAMS.dt, p, GAP_SCHEME,
                     tracked_obs, mi; ps = ps, which_observer = "Live",
                     tracked_times = tracked_times, k_face_choice = K_FACE,
                     verbose = SIM_PARAMS.verbose)

    # Run simulation
    @time run!(sim)

    # Save figures
    if SIM_PARAMS.save_figs
        dir = joinpath(@__DIR__, "Figures", ACTIVE_LABEL)
        mkpath(dir)

        moulin_locations = get_moulin_ij(state)
        make_mp4_2d(sim.observer, "h", moulin_locations; filename = joinpath(dir, "head.mp4"))
        make_mp4_2d(sim.observer, "b", moulin_locations; filename = joinpath(dir, "gap_height.mp4"))
        make_mp4_mid(sim.observer, "b", ceil(Int, NY / 2), moulin_locations; filename = joinpath(dir, "gap_height_midline.mp4"))

        CairoMakie.activate!()
        fig = Figure(size = (1200, 600))
        ax1 = Axis(fig[1, 1], title = "Gap Height (m)", xlabel = "x (m)", ylabel = "y (m)")
        hm_b = CairoMakie.heatmap!(ax1, sim.grid.x, sim.grid.y, Array(sim.state.b))
        Colorbar(fig[1, 2], hm_b)
        ax2 = Axis(fig[1, 3], title = "Head (m)", xlabel = "x (m)", ylabel = "y (m)")
        hm_h = CairoMakie.heatmap!(ax2, sim.grid.x, sim.grid.y, Array(sim.state.h))
        Colorbar(fig[1, 4], hm_h)
        save(joinpath(dir, "final_state.png"), fig)
        println("Saved figures to: $dir")
    end

    println("Finished: $ACTIVE_LABEL")
end

main()
