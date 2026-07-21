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
    # label            => (FT,       backend,    nx,  ny,  gap_scheme,  K_face,       e_v,  alpha,   linear_solver,  mask_choice, b_min, lx,   ly,   tsteps,  dt,     picard_iters, picard_tol)
    "baseline_LU"       => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "LU",          "barrier",   0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "gmres_sparse"      => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "GMRES",       "barrier",   0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "gmres_mf"          => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "GMRES_MF",    "barrier",   0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "bicgstab_sparse"   => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "BiCGSTAB",    "barrier",   0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "bicgstab_mf"       => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "BiCGSTAB_MF", "barrier",   0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "simple_mask"       => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  nothing, "LU",          "simple",    0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "harmonic_K"        => (Float64, "Threads",   32,  32,  "implicit",  "harmonic",   0.0,  0.1,     "LU",          "simple",    0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "implicit_gap"      => (Float64, "Threads",   32,  32,  "implicit",  "arithmetic", 0.0,  nothing, "LU",          "simple",    0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "relaxation"        => (Float64, "Threads",   32,  32,  "explicit",  "arithmetic", 0.0,  0.5,     "LU",          "barrier",   0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "larger_n"          => (Float64, "Threads",   64,  64,  "explicit",  "arithmetic", 0.0,  nothing,     "BiCGSTAB_MF", "simple",    0.0,   1e3,  1e3,  24*24,   3600.0, 500,          1e-6),
    "gpu"               => (Float32, "Metal",     64,  64,  "explicit",  "arithmetic", 0.0,  nothing, "BiCGSTAB_MF", "simple",    0.0,   1e3,  1e3,  10,      3600.0, 500,          1e-3),
)

# ----------------------------------------------------------------------------
# Pick which config to run
# ----------------------------------------------------------------------------
const ACTIVE_LABEL = "larger_n"

haskey(CONFIGS, ACTIVE_LABEL) || error("Unknown config label '$ACTIVE_LABEL'. Choose one of: $(join(keys(CONFIGS), ", "))")
const (FT, BACKEND, NX, NY, GAP_SCHEME, K_FACE, E_V, ALPHA, LINEAR_SOLVER, MASK_CHOICE, B_MIN, LX, LY, TSTEPS, DT, PICARD_ITERS, PICARD_TOL) = CONFIGS[ACTIVE_LABEL]

# Preferences must be set before `using Shakti` to take effect
set_preferences!("Shakti", "backend" => BACKEND, "floattype" => string(FT); force = true)

using Shakti
using CairoMakie

# ============================================================================
# SIMULATION SETTINGS (fixed across configs)
# ============================================================================
const SIM_PARAMS = (
    slope = 0.02,
    ice_thickness = 500.0,
    water_depth = 0.01,
    A_visc_val = 5e-25,
    moulin_flux = 4.0,    # total moulin input (m^3/s), spread over moulin_radius regardless of grid resolution
    moulin_radius = 15.0, # m -- fixed physical footprint, so ieb's intensity doesn't scale with 1/(dx*dy) as the grid is refined
    verbose = true,
    make_videos = true, # head.mp4 / gap_height.mp4 / gap_height_midline.mp4
    make_plot = true,   # final_state.png
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
        # Downstream obstacle that forces flow to split around it, positioned relative to
        # the moulin (im, jm) and sized relative to the grid -- rather than the original
        # hardcoded (20, 15:17), which was only ever valid for the 32x32/moulin-at-(16,16)
        # case and put the barrier somewhere nonsensical (or out of bounds) at other sizes.
        # offset/halfwidth reproduce that original geometry's ratios (4/32 downstream, 1
        # cell either side of jm) so 32x32 is unchanged; clamp keeps it in-bounds and
        # non-degenerate down to small grids.
        offset = clamp(round(Int, 0.125 * nx), 1, nx - im - 1)
        ic = im + offset
        halfwidth = clamp(round(Int, nx / 32), 1, min(jm - 2, ny - jm - 1))
        for jc in (jm - halfwidth):(jm + halfwidth)
            mask[ic, jc] = OTHER_BASIN
        end
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
    println("e_v: $E_V, alpha: $ALPHA, Solver: $LINEAR_SOLVER, Mask: $MASK_CHOICE, b_min: $B_MIN")
    println("lx: $LX, ly: $LY, tsteps: $TSTEPS, dt: $DT, picard_iters: $PICARD_ITERS, picard_tol: $PICARD_TOL")
    println("="^80)

    # Grid and state
    grid = Grid(NX, NY, LX, LY)
    state = State(grid)

    # Model parameters
    p = ModelParameters(e_v = E_V, b_min = B_MIN)

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
    # Spread the moulin input over a fixed physical footprint (moulin_radius) rather
    # than always a single cell -- at fixed total flux, injecting into one cell means
    # ieb = flux / (dx*dy) grows without bound as the grid is refined (a 256x256 grid
    # has 64x smaller cells than 32x32, so the same total flux becomes 64x more locally
    # intense and blows up Picard/the gap-height update within a step or two).
    xm, ym = grid.x[im], grid.y[jm]
    footprint = [(grid.x[i] - xm)^2 + (grid.y[j] - ym)^2 <= SIM_PARAMS.moulin_radius^2 for i in 1:NX, j in 1:NY]
    n_footprint = count(footprint)
    ieb[footprint] .= SIM_PARAMS.moulin_flux / (n_footprint * grid.dx * grid.dy)

    set_initial_conditions!(state, grid, p, mi, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)

    # Solvers
    ls = build_linear_solver(LINEAR_SOLVER, grid)
    ps = PicardSolver(PICARD_ITERS, PICARD_TOL, ls, grid; alpha = ALPHA)

    # Observer and simulation
    tracked_obs = ["h", "b"]
    tracked_times = 0:1:TSTEPS

    sim = Simulation(grid, state, TSTEPS, floattype(DT), p, GAP_SCHEME,
                     tracked_obs, mi; ps = ps, which_observer = "Live",
                     tracked_times = tracked_times, k_face_choice = K_FACE,
                     verbose = SIM_PARAMS.verbose)

    # Run simulation
    @time run!(sim)

    # Save figures
    if SIM_PARAMS.make_videos || SIM_PARAMS.make_plot
        dir = joinpath(@__DIR__, "Figures", ACTIVE_LABEL)
        mkpath(dir)

        if SIM_PARAMS.make_videos
            moulin_locations = get_moulin_ij(state)
            make_mp4_2d(sim.observer, "h", moulin_locations; filename = joinpath(dir, "head.mp4"))
            make_mp4_2d(sim.observer, "b", moulin_locations; filename = joinpath(dir, "gap_height.mp4"))
            make_mp4_mid(sim.observer, "b", ceil(Int, NY / 2), moulin_locations; filename = joinpath(dir, "gap_height_midline.mp4"))
            println("Saved videos to: $dir")
        end

        if SIM_PARAMS.make_plot
            CairoMakie.activate!()
            fig = Figure(size = (1200, 600))
            ax1 = Axis(fig[1, 1], title = "Gap Height (m)", xlabel = "x (m)", ylabel = "y (m)")
            hm_b = CairoMakie.heatmap!(ax1, sim.grid.x, sim.grid.y, Array(sim.state.b))
            Colorbar(fig[1, 2], hm_b)
            ax2 = Axis(fig[1, 3], title = "Head (m)", xlabel = "x (m)", ylabel = "y (m)")
            hm_h = CairoMakie.heatmap!(ax2, sim.grid.x, sim.grid.y, Array(sim.state.h))
            Colorbar(fig[1, 4], hm_h)
            save(joinpath(dir, "final_state.png"), fig)
            println("Saved plot to: $dir")
        end
    end

    println("Finished: $ACTIVE_LABEL")
end

main()
