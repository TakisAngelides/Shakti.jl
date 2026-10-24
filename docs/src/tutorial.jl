#=
# Tutorial

One runnable script, every decision from the grid to the plots, on a synthetic glacier. Each step
lists the options as `Option(...)  what it does, when to pick it`; the live line below the list is
the default or a typical pick. Change it to switch.

Steps: 1 preferences, 2 grid, 3 mask, 4 state, 5 parameters, 6 sliding law and melt input, 7 the
sheet, 8 solvers, 9 gap scheme and time step, 10 output, 11 run and check, 12 coupling, 13 plot.

This page runs when the docs are built: one year of 1-day steps on an 80 x 20 grid, in under a minute.
=#

using Shakti
using NetCDF
using CairoMakie
using Statistics: mean

#=
## 1. Preferences
Set once, before `using Shakti`, then restart Julia (`set_preferences!` writes `LocalPreferences.toml`).

`backend`:
* `"Threads"` (default)  CPU kernels; the only backend the sparse direct solvers (and CUDSS) work with
* `"CUDA"`, `"Metal"`  fields live on the GPU; only `CGIterativeSolver(grid, MatrixFreeLinearSystem)` solves there, so not the default sheet

`floattype`:
* `"Float64"` (default)
* `"Float32"`  halves memory, looser tolerances

`cudss`:
* `true` (default)  load CUDA.jl/CUDSS.jl under `"Threads"`, so `CUDSSDirectSolver` works on a GPU node
* `false`  CPU-only work, saves minutes of load time

Threads: start Julia with `-t N`.
=#
backend, floattype

#=
## 2. Grid
* `Grid(nx, ny, lx, ly)`  regular grid of `nx x ny` cells over `lx x ly` metres; cell centres in `grid.x`, `grid.y`
=#
nx, ny = 80, 20
grid = Grid(nx, ny, 20000.0, 5000.0)

#=
## 3. Mask
One code per cell. It sets the boundary conditions:
* `GROUNDED`  grounded ice, solved
* `LAND`  ice-free land: an outlet at atmospheric pressure
* `OCEAN`  ocean or floating ice: an outlet at the sea-water pressure of the bed
* `OTHER_BASIN`  ice outside the basin or no data: closed, no water crosses
* `FROZEN_BED`  frozen bed: `p_w = 0`, closed (set by `update_frozen_mask!`, step 12)

The grid edge is closed, so every grounded region needs an outlet. Water leaves through `LAND`/`OCEAN`
faces and never enters (`outflow_only_land`, `outflow_only_ocean`, step 5).
=#
mask = fill(GROUNDED, nx, ny)
mask[1, :] .= LAND                  # terminus at x = 0
mask[end, :] .= OTHER_BASIN
mask[:, [1, end]] .= OTHER_BASIN

#=
## 4. State and initial conditions
`set_initial_conditions!` fills a `State` from the geometry and the inputs below, all SI:
* `zb`, `zs`  bed and surface elevation [m]
* `b`  initial gap height [m]; keep it above `b_min` (step 5)
* `A_visc`  ice rate factor [Pa^-3 s^-1]
* `G`  geothermal flux [W/m2]
* `ub_x`, `ub_y`  sliding speed on the x- and y-faces [m/s]
* `ieb`  water input from above [m/s]
* `taub_x`, `taub_y`  basal shear stress on the faces [Pa] (used by `PrescribedSlidingLaw` only)
=#
zb     = zeros(nx, ny)
zs     = [6 * (sqrt(x + 5000.0) - sqrt(5000.0)) + 1 for x in grid.x, y in grid.y]   # SHMIP (de Fleurian et al. 2018)
b      = fill(0.01, nx, ny)
A_visc = fill(5e-25, nx, ny)
G      = fill(0.05, nx, ny)
ub_x   = fill(1e-6, nx + 1, ny)
ub_y   = zeros(nx, ny + 1)
ieb    = fill(1.0 / (365 * 86400.0), nx, ny)   # 1 m/a
taub_x, taub_y = zeros(nx + 1, ny), zeros(nx, ny + 1)

# The geometry along `x` (the slab is uniform in `y`): a flat bed and the ice-sheet margin of SHMIP,
# 1 m thick at the outlet and about 520 m thick 20 km inland.
xg = grid.x ./ 1e3
geo = Figure(size = (720, 280))
axg = Axis(geo[1, 1]; xlabel = L"x\ [\mathrm{km}]", ylabel = L"z\ [\mathrm{m}]",
          xtickformat = vs -> [L"%$(round(Int, v))" for v in vs], ytickformat = vs -> [L"%$(round(Int, v))" for v in vs])
band!(axg, xg, zb[:, 1], zs[:, 1]; color = (:lightblue, 0.6), label = L"\mathrm{ice}")
band!(axg, xg, fill(-100.0, nx), zb[:, 1]; color = (:sienna, 0.6), label = L"\mathrm{bed}")
lines!(axg, xg, zs[:, 1]; color = :steelblue, linewidth = 2)
lines!(axg, xg, zb[:, 1]; color = :saddlebrown, linewidth = 2)
ylims!(axg, -100, 600)
Legend(geo[1, 2], axg; framevisible = false)
geo

#=
## 5. Parameters
`ModelParameters(; ...)`, every field has a default. The ones that change results most:
* `omega = 1e-3`  laminar-turbulent transition (Sommers et al. 2018, Table 2)
* `br = 0.1`, `lr = 2.0`  bed-bump height and spacing [m]; `br = 0` switches opening by sliding off
* `ct = 7.5e-8`  pressure-melting coefficient; `0` drops the sensible-heat term (as ISSM's SHAKTI does)
* `b_min = 1e-6`  lower clamp of `b`; keep it well below the initial `b`, or every cell can sit on the clamp
* `b_max = Inf`  upper clamp; ISSM's SHAKTI uses `1.0`. A finite value hides a gap runaway, `Inf` lets `check_gap_height` report it
* `N_min = -Inf`, `N_max = Inf`  clamps of `N`; off, `IceUplift` (step 7) is the physical bound
* `e_v = 0`  englacial void ratio; `0` elliptic head equation, `> 0` parabolic (the head stores englacial water, see the [Parabolic scheme](@ref ParabolicScheme) page)
* `outflow_only_land = true`, `outflow_only_ocean = true`  outlets let water out, never in; `false` for tidal intrusion
* `b_c = 0`  creep cutoff length (SUHMO); non-zero selects `CreepCutoff`. Dataset-dependent runaways: opt in with care
=#
p = ModelParameters()

#=
## 6. Sliding law and melt input
Sliding law (`tau_b`, which feeds frictional melt):
* `RegularizedCoulombSlidingLaw(C)`  scalar `C`, `lambda = 1.5 A` from the initial `A_visc`
* `RegularizedCoulombFieldSlidingLaw(grid, C)`  `C` a cell field
* `RegularizedCoulombV0SlidingLaw(grid, C, v0; q = 1/3)`  Joughin et al. (2019), `v0` in m/s
* `LinearSlidingLaw(grid, C)`  `tau_b = C^2 N u_b`
* `PrescribedSlidingLaw()`  keep the `taub_x`, `taub_y` given in step 4 (e.g. from an ice-flow model)

Melt input (water from above):
* `ConstantMeltInput()`  the `ieb` of step 4, unchanged
* `SeasonalMeltInput(; t_start, t_end, amplitude, offset, i_min)`  Sommers et al. (2018) Sect. 3.3, see the [Seasonal melt](@ref SeasonalMeltInput) page
* `GaussianMoulinMeltInput(grid, x0, y0, sigma, Q_max, ramp_duration)`  one moulin [m3/s], ramped up
* a coupled model adds its own with `set_ieb_external!` (step 12)
=#
sl = RegularizedCoulombSlidingLaw(0.25)
mi = ConstantMeltInput()

state = State(grid)
set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

#=
## 7. The sheet
The defaults are the GlaDS-2 sheet. Each keyword of `Simulation` has an opt-out:
* `cavity_filling = UnfilledCavities()` (default)  a head below the bed empties the cavity, `p_w >= 0`
  * `FilledCavities()`  SHAKTI's always-full gap; `p_w < 0` (suction) appears where the gap cannot hold the head
* `uplift = IceUplift()` (default)  water above the overburden lifts the ice, `p_w <= p_o`
  * `NoUplift()`  SHAKTI's, `N < 0` allowed
* `hydraulic_potential = SheetTopPotential()` (default)  `q = -K grad(h + b)`
  * `BedPotential()`  `q = -K grad(h)`, the original equations
* `k_face_choice = "implicit_upwind"` (default with the two above)  face conductance from the upwind water depth at the end of the step: no time-step limit, needs an LU solver (step 8)
  * `"upwind"`  symmetric matrix (Cholesky, CG, Newton allowed), but stable only for `dt < dx W / (3|q|)` (minutes at 100 m); warns with unfilled cavities or uplift
  * `"arithmetic"` (default with `FilledCavities`/`NoUplift`), `"harmonic"`  averaged faces; refused with unfilled cavities or uplift, where they create water
* `gap_coupling = ConsistentGapCoupling()` (default with a Picard solver)  the head equation uses the gap change the gap scheme makes: the water budget closes
  * `ExplicitGapCoupling()`  SHAKTI's: the head equation assumes `dt*rate(N, b_old)`. Creates water where `b` is clamped (13 times the melt on Drang Drung) and can let the gap run away at `N < 0`; use it only to reproduce SHAKTI/ISSM

Which to choose:
* default: physically bounded `p_w`, closed budget, large steps (hours to days)
* SHAKTI's sheet: `cavity_filling = FilledCavities(), uplift = NoUplift()`, and Cholesky is allowed
* the original equations bit for bit: also `hydraulic_potential = BedPotential(), gap_coupling = ExplicitGapCoupling()`
=#

#=
## 8. Solvers
Head solve, `ps`, for any `e_v`:
* `PicardSolver(iters, tol, ls, grid; alpha = nothing, atol = 0)`  fixed-point iteration; `alpha < 1` under-relaxes a stalling solve, `atol > 0` [m] replaces the relative test

Linear solver `ls`:
* `LUDirectSolver(grid)`  UMFPACK LU, CPU; the default sheet's matrix is non-symmetric
* `CUDSSDirectSolver(grid)` (`structure = :general`, the default)  the same on a GPU node (fields stay on the CPU); the fastest on large grids
* `CholeskyDirectSolver(grid)`, `CUDSSDirectSolver(grid; structure = :spd)`  symmetric matrices only: SHAKTI's sheet or `"upwind"`
* `CGIterativeSolver(grid, SparseAssembledLinearSystem; amg = true)`  symmetric only, CPU
* `CGIterativeSolver(grid, MatrixFreeLinearSystem)`  symmetric only, the one GPU-resident solver

`Simulation` refuses Cholesky and CG with the default sheet, and names the opt-outs.
=#
ps = PicardSolver(200, 1e-6, LUDirectSolver(grid), grid)

#=
## 9. Gap scheme and time step
`gap_scheme` (keyword):
* `FullyImplicitGapScheme()` (default)  unconditionally stable for any bump height
* `ImplicitGapScheme()`  implicit creep, explicit opening; deprecated (warns)
* `ExplicitGapScheme()`  forward Euler; small steps only

Older drivers name it by a string before the tracked observables (`"fully_implicit"`, `"implicit"`,
`"explicit"`); that form still works.

Under `WithDiffusion` the choice is ignored: the diffusive gap solve has its own scheme.

Time step:
* `FixedTimeStep()` (default)  `dt` every step; with the default sheet hours to days work
* `AdaptiveTimeStep(grid, state; safety_factor = 0.3, dt_min, dt_max, target_time)`  `dt` from the creep and sliding rates; `run!` stops at `target_time`
=#
dt, nsteps = 86400.0, 365

#=
## 10. Output
* `which_observer = "IO"` with `which_file_writer = "NetCDF"` (or `"HDF5"`, `"JLD2"`)  fields written at `tracked_times` (step indices, 0 = initial state); use this for long runs
* `"CSV"`  min, max and mean per field and time only
* `"Live"`  kept in memory, for short interactive runs
* `tracked_obs`  any `State` field names, e.g. `"N"`, `"b"`, `"h"`, `"Re"`, `"q_x"`, `"b_empty"`, `"b_uplift"`

Checkpoints: `run!(sim; checkpoint_every, checkpoint_path)`, then `restart_path` resumes a killed run
and `extend_path` continues a finished one into a new file.
=#
path = joinpath(mktempdir(), "tutorial.nc")
sim = Simulation(grid, state, nsteps, floattype(dt), p,
                 ["N", "b", "Re", "po"], mi, sl; ps = ps,
                 which_observer = "IO", which_file_writer = "NetCDF",
                 tracked_times = 0:nsteps, path = path)

#=
## 11. Run and check
`run!` warns once if the gap grows taller than the ice (`check_gap_height`), the sign of a gap
runaway. `verbose = true` in `Simulation` prints each step's Picard iterations.
=#
run!(sim)

gr = Array(state.mask) .== GROUNDED
(converged = sim.hs.ps.converged,
 N_over_po = extrema(Array(state.N)[gr] ./ Array(state.po)[gr]),
 b = extrema(Array(state.b)[gr]))

#=
## 12. Coupling to an ice-flow model
Between steps (`step!(sim)` advances one) a host model can:
* change the geometry or sliding speed in `state` (`zs`, `zb`, `ub_x`, `ub_y`), refreshed at the next step
* `set_mask!(state, p, new_mask)`  follow its grounding line and ice extent
* `update_frozen_mask!(state, p, T_prime_b)`  freeze and thaw cells from the basal temperature
* `set_ieb_external!(state, ieb_external)`  add its englacial or surface input [m/s]
* `freeze_on_capacity!(C, sim; dt)`  the water the bed can freeze on in `dt`, for its thermal model

`state.N` is the effective pressure the sliding law needs.
=#

#=
## 13. Plot
The slab is uniform in `y`, so a profile along `x` says it all: the effective pressure, the gap
height and the Reynolds number on days 0, 1, 10, 30, 100 and 365, from the NetCDF file.
* `N` goes to zero at the outlet, where the ice is 1 m thick and the water drains at atmospheric
  pressure; `N/p_o` there is close to 1.
* `N` is set within a day. The gap takes about 100 days: it opens near the outlet, where the
  flux is largest and the thin ice closes it slowly (closure goes as `A N^3 b`), and closes inland.
* `N` and `Re` start on day 1: on day 0 they come from the starting head, not from a solve.
=#
N, bb, Re = (ncread(path, v) for v in ("N", "b", "Re"))
days = round.(Int, ncread(path, "time") ./ 86400)
frames = [1, 2, 11, 31, 101, nsteps + 1]             # days 0, 1, 10, 30, 100, 365
jm = ny ÷ 2                                          # one row along x
ix = findall(gr[:, jm])                              # grounded cells only
xkm = grid.x[ix] ./ 1e3
lin = vs -> [L"%$(round(v, digits = 1))" for v in vs]
pow10 = vs -> [L"10^{%$(round(Int, log10(v)))}" for v in vs]
decades(v) = 10.0 .^ (floor(Int, log10(minimum(v))):ceil(Int, log10(maximum(v))))   # log ticks at integer powers of ten
colors = cgrad(:viridis, length(frames); categorical = true)

fig = Figure(size = (720, 640))
rows = [(L"N\ [\mathrm{MPa}]", k -> N[ix, jm, k] ./ 1e6, identity, lin, frames[2:end]),
        (L"b\ [\mathrm{mm}]", k -> bb[ix, jm, k] .* 1e3, identity, lin, frames),
        (L"\mathrm{Re}", k -> Re[ix, jm, k], log10, pow10, frames[2:end])]
for (r, (name, f, scale, fmt, ks)) in enumerate(rows)
    yticks = scale === log10 ? decades(reduce(vcat, f.(ks))) : Makie.automatic
    ax = Axis(fig[r, 1]; ylabel = name, yscale = scale, yticks, ytickformat = fmt,
              xtickformat = vs -> [L"%$(round(Int, v))" for v in vs],
              xlabel = r == 3 ? L"x\ [\mathrm{km}]" : "", xticklabelsvisible = r == 3)
    for (c, k) in enumerate(frames)
        k in ks || continue
        lines!(ax, xkm, f(k); color = colors[c], linewidth = 2, label = L"\mathrm{day}\ %$(days[k])")
    end
    r == 2 && Legend(fig[1:3, 2], ax; framevisible = false)
end
rowgap!(fig.layout, 8)
fig
