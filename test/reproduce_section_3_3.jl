import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Preferences

# ============================================================================
# Reproduces Sect. 3.3 ("Seasonal variation and distributed meltwater input")
# of Sommers, Rajaram & Morlighem (2018), GMD 11, 2955-2974.
# https://doi.org/10.5194/gmd-11-2955-2018
#
# Setup, straight from the paper (Table 2 and Sect. 3.3 text):
#   - 4 km (x) by 8 km (y) domain, flat bed (zb = 0 everywhere).
#   - Ice surface parabolic in x, uniform in y: thickness 550 m at x=0 up to
#     700 m at x=4 km. The paper gives the endpoints but not the formula; we
#     use H(x) = sqrt(H0^2 + (H1^2 - H0^2)*x/Lx), the standard "H^2 linear in
#     x" convention for parabolic glacier profiles (Vialov/perfect-plasticity
#     shape) used throughout the GlaDS/SHAKTI literature (e.g. Werder et al.
#     2013's synthetic test geometries) -- override via --thickness-min/max
#     if you have the exact ISSM tutorial formula.
#   - Outflow at x=0 (LAND: Dirichlet h = zb = 0, i.e. atmospheric pressure,
#     paper's Eq. 14); the other three edges are zero-flux (OTHER_BASIN,
#     paper's Eq. 15 with f=0).
#   - Meltwater input distributed uniformly over the whole domain (not
#     moulins): steady 1 m/a during spin-up, then SeasonalMeltInput's Eq. 16
#     cosine cycle for 1 year.
#   - Sliding velocity, geothermal flux, bed bump geometry, flow-law exponent
#     etc. all from Table 2. The flow-law rate factor A is left blank in
#     Table 2 (not given anywhere in the paper's text either); we default to
#     5e-25 Pa^-3 s^-1, matching this repo's existing convention in
#     options_explorer.jl -- override via --A if you have the real value.
#   - Zero englacial storage (e_v = 0, stated explicitly in Sect. 3.3).
#   - Explicit gap-height time stepping (the paper's Sect. 2.3: implicit
#     backward Euler for h, explicit for b -- this isn't a per-example
#     choice, it's how the whole model is formulated).
#   - Initial gap height 0.01 m + N(0, 1%) noise (paper's Sect. 3.3).
# ============================================================================

# ----------------------------------------------------------------------------
# CLI args (--key value pairs; run `julia reproduce_section_3_3.jl --help`)
# ----------------------------------------------------------------------------
const DEFAULTS = (
    nx = 32, ny = 32,                      # grid resolution (Cholesky-sized)
    lx = 4000.0, ly = 8000.0,              # domain size, m (paper: 4 km x 8 km)
    thickness_min = 550.0, thickness_max = 700.0, # ice thickness at x=0 / x=lx, m
    A = 5e-25,                             # flow-law rate factor, Pa^-3 s^-1 (NOT given by the paper -- see note above)
    dt = 3600.0,                           # s (paper: 1 h throughout)
    spinup_days = 4.0,                     # paper: steady state reached in 4 days
    run_days = 365.0,                      # paper: 1 full annual cycle
    picard_iters = 500, picard_tol = 1e-6,
    alpha = 0.1,                           # Picard under-relaxation; the paper's own Fig. 10 shows this Picard/dt=1h combination oscillates and fails to converge once channelization onsets (their fix was shrinking dt; this is the equivalent stabilizer on the iteration itself)
    omega = 0.0001,                         # Table 2's value; controls the laminar/turbulent transition (Eq. 5) -- lower moves a given Re further into the turbulent branch instead of sitting at omega*Re~O(1), the numerically hard transitional zone
    sliding_law = "coulomb",               # "coulomb" (RegularizedCoulombSlidingLaw, tau_b = f(N, u_b, C)) or "driving-stress" (PrescribedSlidingLaw, tau_b = rho_i*g*H*|surface slope|, parameter-free). The paper's own Table 2 has NO friction coefficient at all -- tau_b isn't part of SHAKTI's own equations, see melt_rate.jl's AbstractSlidingLaw note -- so neither option is literally "the paper's choice"; this defaults to the codebase's prior convention.
    C = 0.25,                              # Coulomb friction coefficient, only used when sliding_law="coulomb" (NOT given by the paper -- see note above)
    seed = 1,                              # RNG seed for the initial gap-height perturbation
    out_dir = joinpath(@__DIR__, "Figures", "section_3_3"),
)

function parse_cli(args, defaults)
    if "--help" in args || "-h" in args
        println("Usage: julia reproduce_section_3_3.jl [--key value ...]")
        println("Keys (defaults shown): ", join(("--$k $v" for (k, v) in pairs(defaults)), "  "))
        exit(0)
    end
    opts = Dict{Symbol, Any}(pairs(defaults))
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("Unexpected argument: $arg")
        key = Symbol(replace(arg[3:end], "-" => "_"))
        haskey(opts, key) || error("Unknown option: $arg. Choose from: $(join(("--"*string(k) for k in keys(defaults)), ", "))")
        i += 1
        i <= length(args) || error("Missing value for $arg")
        default = defaults[key]
        opts[key] = default isa Int ? parse(Int, args[i]) :
                    default isa Float64 ? parse(Float64, args[i]) :
                    args[i]
        i += 1
    end
    return NamedTuple(opts)
end

const OPTS = parse_cli(ARGS, DEFAULTS)

set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using CairoMakie
using Random
using Statistics

function main()

    println("="^80)
    println("Reproducing SHAKTI paper Sect. 3.3: $(OPTS.nx)x$(OPTS.ny), A=$(OPTS.A), spinup=$(OPTS.spinup_days)d, run=$(OPTS.run_days)d")
    println("="^80)

    NX, NY = OPTS.nx, OPTS.ny
    grid  = Grid(NX, NY, OPTS.lx, OPTS.ly)
    state = State(grid)

    # Zero englacial storage (paper, Sect. 3.3) -- also ModelParameters' default.
    p = ModelParameters(e_v = 0.0, b_min = 1e-3, omega = OPTS.omega)

    # ------------------------------------------------------------------------
    # Mask: LAND outflow at x=0 (Dirichlet h=zb=0), zero-flux (OTHER_BASIN)
    # on the other three edges (paper's Eqs. 14-15).
    # ------------------------------------------------------------------------
    mask = fill(GROUNDED, NX, NY)
    mask[1, :]   .= LAND
    mask[end, :] .= OTHER_BASIN
    mask[:, 1]   .= OTHER_BASIN
    mask[:, end] .= OTHER_BASIN

    # ------------------------------------------------------------------------
    # Geometry: flat bed, parabolic-in-x surface (see module docstring above)
    # ------------------------------------------------------------------------
    zb = zeros(NX, NY)
    H0, H1 = OPTS.thickness_min, OPTS.thickness_max
    Hx = sqrt.(H0^2 .+ (H1^2 - H0^2) .* (grid.x ./ OPTS.lx))
    zs = repeat(Hx, 1, NY) # zb=0, so surface elevation == thickness

    A_visc = fill(OPTS.A, NX, NY)
    b      = fill(0.01, NX, NY)
    G      = fill(0.05, NX, NY)   # Table 2
    ub_x   = fill(1e-6, NX + 1, NY) # Table 2: 1e-6 m/s (31.5 m/a), uniform, toward the x=0 outflow
    ub_y   = zeros(NX, NY + 1)

    # ------------------------------------------------------------------------
    # Sliding law: see melt_rate.jl's AbstractSlidingLaw note -- the paper
    # doesn't specify how tau_b is obtained for these standalone examples.
    # ------------------------------------------------------------------------
    if OPTS.sliding_law == "coulomb"
        sl = RegularizedCoulombSlidingLaw(OPTS.C)
        taub_x = zeros(NX + 1, NY) # unused: recomputed from N/ub every Picard iteration
        taub_y = zeros(NX, NY + 1)
    elseif OPTS.sliding_law == "driving-stress"
        sl = PrescribedSlidingLaw()
        # tau_b = rho_i*g*H*|surface slope|, evaluated on x-faces the same way
        # the model differences h/pw internally (interior faces only; the two
        # domain-boundary faces are left at zero, undefined at the edge).
        # zs is uniform in y (Sect. 3.3), so there's no y-direction driving
        # stress, consistent with ub_y = 0.
        taub_x = zeros(NX + 1, NY)
        for i in 2:NX
            dHdx = (Hx[i] - Hx[i-1]) / grid.dx
            Hface = (Hx[i] + Hx[i-1]) / 2
            taub_x[i, :] .= p.rho_i * p.g * Hface * abs(dHdx)
        end
        taub_y = zeros(NX, NY + 1)
    else
        error("Unknown sliding_law: \"$(OPTS.sliding_law)\" (expected \"coulomb\" or \"driving-stress\")")
    end

    # Initial gap height: 0.01 m + N(0, 1%) noise (Sect. 3.3), to seed
    # channelization instabilities.
    rng = Random.MersenneTwister(OPTS.seed)
    b .*= 1 .+ 0.01 .* randn(rng, NX, NY)

    seconds_per_year = 365 * 86400.0

    # ------------------------------------------------------------------------
    # Phase 1: spin-up with steady 1 m/a distributed input
    # ------------------------------------------------------------------------
    mi_spinup = ConstantMeltInput()
    ieb_spinup = fill(1.0 / seconds_per_year, NX, NY) # 1 m/a -> m/s

    set_initial_conditions!(state, grid, p, mi_spinup, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb_spinup, taub_x, taub_y)

    ls = CholeskyDirectSolver(grid)
    ps = PicardSolver(OPTS.picard_iters, OPTS.picard_tol, ls, grid; alpha = OPTS.alpha)

    spinup_tsteps = round(Int, OPTS.spinup_days * 86400 / OPTS.dt)
    sim_spinup = Simulation(grid, state, spinup_tsteps, floattype(OPTS.dt), p, "implicit",
                             String[], mi_spinup, sl; ps = ps, verbose = true)

    println("--- Spin-up: $spinup_tsteps steps ($(OPTS.spinup_days) days) ---")
    @time run!(sim_spinup)

    # ------------------------------------------------------------------------
    # Phase 2: seasonal cycle, Eq. 16, for 1 year -- reuses the spun-up state
    # and the same factorized Cholesky solver (mask/topology unchanged).
    # ------------------------------------------------------------------------
    mi_seasonal = SeasonalMeltInput()
    initialize_ieb!(mi_seasonal, state, ieb_spinup) # harmless: overwritten by update_ieb! on the first step below

    run_tsteps = round(Int, OPTS.run_days * 86400 / OPTS.dt)
    tracked_obs = ["h", "b", "N"]
    tracked_times = 0:1:run_tsteps # dense: needed for the min/mean/max time series below

    sim_seasonal = Simulation(grid, state, run_tsteps, floattype(OPTS.dt), p, "implicit",
                               tracked_obs, mi_seasonal, sl; ps = ps, which_observer = "Live",
                               tracked_times = tracked_times, verbose = true)

    println("--- Seasonal cycle: $run_tsteps steps ($(OPTS.run_days) days) ---")
    @time run!(sim_seasonal)

    # ------------------------------------------------------------------------
    # Plots: Fig. 5-style (min/mean/max gap height & head vs. day) and
    # Fig. 6-style (gap height / head / effective pressure snapshots).
    # ------------------------------------------------------------------------
    mkpath(OPTS.out_dir)
    hist = sim_seasonal.observer.history
    days = (0:run_tsteps) .* (OPTS.dt / 86400)

    b_hist, h_hist = hist["b"], hist["h"]
    b_min = [minimum(view(b_hist, :, :, i)) for i in axes(b_hist, 3)]
    b_mean = [mean(view(b_hist, :, :, i)) for i in axes(b_hist, 3)]
    b_max = [maximum(view(b_hist, :, :, i)) for i in axes(b_hist, 3)]
    h_min = [minimum(view(h_hist, :, :, i)) for i in axes(h_hist, 3)]
    h_mean = [mean(view(h_hist, :, :, i)) for i in axes(h_hist, 3)]
    h_max = [maximum(view(h_hist, :, :, i)) for i in axes(h_hist, 3)]

    CairoMakie.activate!()

    fig_ts = Figure(size = (900, 600))
    ax_b = Axis(fig_ts[1, 1], title = "Gap height (m)", xlabel = "Time (d)")
    lines!(ax_b, days, b_min, label = "Min")
    lines!(ax_b, days, b_mean, label = "Mean")
    lines!(ax_b, days, b_max, label = "Max")
    axislegend(ax_b)
    ax_h = Axis(fig_ts[2, 1], title = "Head (m)", xlabel = "Time (d)")
    lines!(ax_h, days, h_min, label = "Min")
    lines!(ax_h, days, h_mean, label = "Mean")
    lines!(ax_h, days, h_max, label = "Max")
    axislegend(ax_h)
    save(joinpath(OPTS.out_dir, "seasonal_timeseries.png"), fig_ts)

    snapshot_days = sort(unique(clamp.([1, 150, 180, 200, 220, 250, 280, round(Int, OPTS.run_days)], 1, round(Int, OPTS.run_days))))
    snapshot_idx = [clamp(round(Int, d * 86400 / OPTS.dt), 0, run_tsteps) + 1 for d in snapshot_days]

    N_hist = hist["N"]
    fig_snap = Figure(size = (250 * length(snapshot_days), 700))
    for (col, (d, idx)) in enumerate(zip(snapshot_days, snapshot_idx))
        ax1 = Axis(fig_snap[1, col], title = "Day $d\ngap height")
        heatmap!(ax1, grid.x, grid.y, view(b_hist, :, :, idx))
        ax2 = Axis(fig_snap[2, col], title = "head")
        heatmap!(ax2, grid.x, grid.y, view(h_hist, :, :, idx))
        ax3 = Axis(fig_snap[3, col], title = "eff. pressure")
        heatmap!(ax3, grid.x, grid.y, view(N_hist, :, :, idx))
    end
    save(joinpath(OPTS.out_dir, "seasonal_snapshots.png"), fig_snap)

    println("Saved figures to: $(OPTS.out_dir)")
    println("Finished.")

end

main()
