# Shakti.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://TakisAngelides.github.io/Shakti.jl/dev/)
[![](https://img.shields.io/badge/license-GNU_GPL_3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![Build Status](https://github.com/TakisAngelides/Shakti.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TakisAngelides/Shakti.jl/actions/workflows/CI.yml?query=branch%3Amain)

Shakti.jl is a Julia implementation of the transient subglacial hydrology model Shakti of
[Sommers et al. (2018)](https://doi.org/10.5194/gmd-11-2955-2018). On a regular grid it evolves the
water-filled gap between ice and bed and gives the hydraulic head, the water flux and the effective
pressure at the ice base. One set of equations covers both distributed (laminar) and channelized
(turbulent) drainage, and which one forms is decided by the solution dynamically.

By default the sheet follows [GlaDS-2](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4310/) physics: 
cavities can be partly empty and water above the ice overburden pressure
lifts the ice, so the water pressure stays between 0 and the ice overburden pressure. Shakti's original sheet
model can be selected with keywords, shown in the
[tutorial](https://TakisAngelides.github.io/Shakti.jl/dev/tutorial/) and run in the
[Helheim](https://TakisAngelides.github.io/Shakti.jl/dev/examples/Helheim/) and
[Drang Drung](https://TakisAngelides.github.io/Shakti.jl/dev/examples/DrangDrung/) examples.
Channel-wall diffusion from Suhmo is available as an option.

## Aim

The aim is a finite differences Julia implementation of the Shakti model, with optional extensions of the physics, that is fast enough to be coupled to an ice-flow model such as
[Yelmo](https://github.com/fesmc/yelmo) over long simulations, on CPU and GPU.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/TakisAngelides/Shakti.jl")
```

The backend (`"Threads"`, `"CUDA"`, `"Metal"`) and float type are set once, before `using Shakti`:

```julia
using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)
```

## Quick start

```julia
using Shakti

# A 20 km x 5 km ice-sheet margin (SHMIP geometry) on an 80 x 20 grid, draining at x = 0.
nx, ny = 80, 20
grid = Grid(nx, ny, 20000.0, 5000.0)

mask = fill(GROUNDED, nx, ny)       # grounded ice, solved
mask[1, :] .= LAND                  # outlet: water leaves at atmospheric pressure
mask[end, :] .= OTHER_BASIN         # the other edges are closed
mask[:, [1, end]] .= OTHER_BASIN

zb     = zeros(nx, ny)                                         # bed elevation [m]
zs     = [6 * (sqrt(x + 5000.0) - sqrt(5000.0)) + 1 for x in grid.x, y in grid.y] # ice surface elevation [m]
b      = fill(0.01, nx, ny)                                    # initial gap height [m]
A_visc = fill(5e-25, nx, ny)                                   # ice rate factor [Pa^-3 s^-1]
G      = fill(0.05, nx, ny)                                    # geothermal flux [W/m2]
ub_x   = fill(1e-6, nx + 1, ny)                                # sliding velocity on x-faces [m/s]
ub_y   = zeros(nx, ny + 1)                                     # sliding velocity on y-faces [m/s]
ieb    = fill(1.0 / (365 * 86400.0), nx, ny)                   # water input from above (e.g. moulin) [m/s] (1 m/a)
taub_x, taub_y = zeros(nx + 1, ny), zeros(nx, ny + 1)          # basal shear stress (recomputed by the sliding law)

p  = ModelParameters() # both physical parameter choices and some numerical choices
sl = RegularizedCoulombSlidingLaw(0.25) # sliding law
state = State(grid) # holds the physical fields defined on the grid e.g. nx, ny center fields, nx+1, ny x-face fields and nx, ny+1 y-face fields
set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y) # set initial conditions for the physical fields

# 10 days of 1 h steps. The head is solved by Picard iteration with a sparse LU solver
# (CUDSSDirectSolver(grid; structure = :general) instead of LUDirectSolver(grid) to run on GPU).
ps  = PicardSolver(200, 1e-6, LUDirectSolver(grid), grid)
sim = Simulation(grid, state, 240, floattype(3600.0), p, # tsteps = 240, dt = 3600 (gap_scheme = FullyImplicitGapScheme() and sheet = GlaDS2Sheet() by default)
                 ConstantMeltInput(), sl; ps = ps, # type of englacial to subglacial melt input, sliding law, which picard solver
                 observer = IOObserver(["h", "N", "b"], 0:24:240, NetCDFFileWriter(), "output.nc")) # write h, N, b every 24 steps to a file
run!(sim)

state.N   # effective pressure [Pa]
state.b   # gap height [m]
state.h   # hydraulic head [m]
```

Every option, and when to choose it, is explained in the
[tutorial](https://TakisAngelides.github.io/Shakti.jl/dev/tutorial/). Runs on real glaciers and ice
sheets are in the [examples](https://TakisAngelides.github.io/Shakti.jl/dev/).

## License

GNU General Public License v3.0 -- see [LICENSE](LICENSE).
