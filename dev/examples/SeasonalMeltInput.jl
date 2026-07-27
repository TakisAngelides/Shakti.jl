#=
# [Seasonal melt input](@id SeasonalMeltInput)
A small, fast-running version of Sect. 3.3 of Sommers, Rajaram & Morlighem (2018)
(https://doi.org/10.5194/gmd-11-2955-2018), "Seasonal variation and distributed meltwater
input": a synthetic slab geometry, spun up under a steady winter melt rate, then driven for
one year by [`SeasonalMeltInput`](@ref)'s cosine-shaped melt-season cycle -- showing how gap
height and hydraulic head respond as the system channelizes and relaxes with the seasons.
=#

using Shakti
using CairoMakie
using Random
using Statistics

# ## Geometry
# A small synthetic domain: flat bed, ice surface parabolic in `x` (thin near the outflow at
# `x=0`, thickening upglacier), uniform in `y` -- the same synthetic setup the paper uses for
# this experiment (Table 2), just at a coarser grid/timestep so this example runs quickly.

const NX, NY = 16, 16
const LX, LY = 4000.0, 8000.0 # 4 km x 8 km, matching the paper's domain
grid = Grid(NX, NY, LX, LY)

zb = zeros(NX, NY)
H0, H1 = 550.0, 700.0 # ice thickness at x=0 / x=Lx (m), the paper's endpoints
Hx = sqrt.(H0^2 .+ (H1^2 - H0^2) .* (grid.x ./ LX)) # zb=0, so surface elevation == thickness
zs = repeat(Hx, 1, NY)

fig_geom = let
    fig = Figure(size = (450, 350))
    ax = Axis(fig[1, 1], title = "Ice surface elevation (m)", xlabel = "x (m)", ylabel = "y (m)")
    hm = heatmap!(ax, grid.x, grid.y, zs)
    Colorbar(fig[1, 2], hm)
    fig
end
fig_geom

# ## Mask and boundary conditions
# Outflow at `x=0` ([`LAND`](@ref): Dirichlet head = 0, i.e. atmospheric pressure); the other
# three edges are zero-flux ([`OTHER_BASIN`](@ref)).

mask = fill(GROUNDED, NX, NY)
mask[1, :]   .= LAND
mask[end, :] .= OTHER_BASIN
mask[:, 1]   .= OTHER_BASIN
mask[:, end] .= OTHER_BASIN

# ## Physical parameters and initial gap height
# Zero englacial storage (`e_v = 0`, required by [`EllipticHeadScheme`](@ref), and the paper's
# own choice too). Initial gap height 0.01 m plus 1% noise, to seed channelization instabilities.
# `omega = 1e-4` (also [`ModelParameters`](@ref)'s own default, deliberately differing from Table
# 2's `0.001` -- see its docstring) keeps the Reynolds numbers reached during channelization out of
# the flux law's steepest, hardest-to-converge transitional band, which is what keeps the Picard
# solve (and hence this time series) stable instead of oscillating; set explicitly here since it's
# what actually makes this example's plots clean, not just an incidental default.

p = ModelParameters(e_v = 0.0, b_min = 1e-3, omega = 1e-4)

A_visc = fill(5e-25, NX, NY)
G      = fill(0.05, NX, NY)
ub_x   = fill(1e-6, NX + 1, NY)
ub_y   = zeros(NX, NY + 1)
sl     = RegularizedCoulombSlidingLaw(0.25)
taub_x = zeros(NX + 1, NY) # unused: RegularizedCoulombSlidingLaw recomputes taub from N/ub every Picard iteration
taub_y = zeros(NX, NY + 1)

rng = Random.MersenneTwister(1)
b = fill(0.01, NX, NY) .* (1 .+ 0.01 .* randn(rng, NX, NY))

# ## Spin-up
# A short spin-up under a steady 1 m/yr distributed melt input, reaching a reasonable starting
# state before switching on the seasonal cycle.

const SECONDS_PER_YEAR = 365 * 86400.0
mi_spinup = ConstantMeltInput()
ieb_spinup = fill(1.0 / SECONDS_PER_YEAR, NX, NY) # 1 m/yr -> m/s

state = State(grid)
set_initial_conditions!(state, grid, p, mi_spinup, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb_spinup, taub_x, taub_y)

ls = CholeskyDirectSolver(grid)
ps = PicardSolver(500, 1e-6, ls, grid; alpha = 0.1) # under-relaxed: the paper's own Fig. 10 shows this Picard/dt combination oscillates once channelization onsets

sim_spinup = Simulation(grid, state, 20, floattype(3600.0), p, "implicit", String[], mi_spinup, sl; ps = ps)
run!(sim_spinup)

# ## Seasonal cycle
# Switch to [`SeasonalMeltInput`](@ref)'s cosine-shaped melt season, and run for one year at the
# paper's own hourly timestep -- `PicardSolver`'s under-relaxation (`alpha = 0.1` above) keeps this
# stable, but a coarser `dt` still lets oscillations creep in around channelization onset (paper's
# Fig. 10), which is what a coarser timestep here previously looked like in the resulting plots.

mi_seasonal = SeasonalMeltInput()
initialize_ieb!(mi_seasonal, state, ieb_spinup) # harmless placeholder value, overwritten on the first step below

dt = 3600.0
tsteps = round(Int, SECONDS_PER_YEAR / dt)
tracked_times = 0:tsteps

sim = Simulation(grid, state, tsteps, floattype(dt), p, "implicit", ["h", "b", "N"], mi_seasonal, sl;
                 ps = ps, which_observer = "Live", tracked_times = tracked_times)
run!(sim)

# ## Results
# Domain min/mean/max gap height and head over the year: both stay low through winter, rise as
# the melt season (roughly days 146-255, `t_start=0.4` to `t_start+period=0.7` of the year)
# drives the system toward a more channelized, higher-flux state, then relax back down.

hist = sim.observer.history
days = (0:tsteps) .* (dt / 86400)
b_hist, h_hist, N_hist = hist["b"], hist["h"], hist["N"]

fig_ts = Figure(size = (650, 500))
ax_b = Axis(fig_ts[1, 1], title = "Gap height (m)", xlabel = "Day of year")
lines!(ax_b, days, [minimum(view(b_hist, :, :, i)) for i in axes(b_hist, 3)], label = "min")
lines!(ax_b, days, [mean(view(b_hist, :, :, i)) for i in axes(b_hist, 3)], label = "mean")
lines!(ax_b, days, [maximum(view(b_hist, :, :, i)) for i in axes(b_hist, 3)], label = "max")
axislegend(ax_b)
ax_h = Axis(fig_ts[2, 1], title = "Hydraulic head (m)", xlabel = "Day of year")
lines!(ax_h, days, [minimum(view(h_hist, :, :, i)) for i in axes(h_hist, 3)], label = "min")
lines!(ax_h, days, [mean(view(h_hist, :, :, i)) for i in axes(h_hist, 3)], label = "mean")
lines!(ax_h, days, [maximum(view(h_hist, :, :, i)) for i in axes(h_hist, 3)], label = "max")
axislegend(ax_h)
fig_ts

# ## Spatial snapshots
# Gap height, head, and effective pressure at representative days across the year: the channel
# the melt season carves out along the outflow direction (`x=0`) opens as the season peaks
# (days ~180-250) and closes back down to the winter baseline by day 365.

grounded = mask .== GROUNDED
mask_nan(field) = ifelse.(grounded, field, NaN)

snapshot_days = (1, 150, 180, 200, 220, 250, 280, 365)
snapshot_idx = [clamp(round(Int, d * 86400 / dt), 0, tsteps) + 1 for d in snapshot_days]

fig_snap = Figure(size = (240 * length(snapshot_days), 650))
for (col, (d, idx)) in enumerate(zip(snapshot_days, snapshot_idx))
    ax1 = Axis(fig_snap[1, col], title = "Day $d\ngap height (m)", aspect = DataAspect())
    heatmap!(ax1, grid.x, grid.y, mask_nan(view(b_hist, :, :, idx)), colormap = :inferno, nan_color = :transparent)
    ax2 = Axis(fig_snap[2, col], title = "head (m)", aspect = DataAspect())
    heatmap!(ax2, grid.x, grid.y, mask_nan(view(h_hist, :, :, idx)), colormap = :dense, nan_color = :transparent)
    ax3 = Axis(fig_snap[3, col], title = "eff. pressure (Pa)", aspect = DataAspect())
    heatmap!(ax3, grid.x, grid.y, mask_nan(view(N_hist, :, :, idx)), colormap = :viridis, nan_color = :transparent)
end
fig_snap
