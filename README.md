# Shakti.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://TakisAngelides.github.io/Shakti.jl/dev/)
[![](https://img.shields.io/badge/license-GNU_GPL_3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![Build Status](https://github.com/TakisAngelides/Shakti.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TakisAngelides/Shakti.jl/actions/workflows/CI.yml?query=branch%3Amain)

Shakti.jl is a Julia implementation of the transient subglacial hydrology model Shakti of
[Sommers et al. (2018)](https://doi.org/10.5194/gmd-11-2955-2018). On a regular grid it evolves the
water layer between ice and bed and gives the hydraulic head, the water flux and the effective
pressure at the ice base. One set of equations covers both distributed (laminar) and channelized
(turbulent) drainage, and which one forms is decided by the solution dynamicallyis .

## Aim

The aim is a Shakti that is fast enough to be coupled to an ice-flow model such as
[Yelmo](https://github.com/palma-ice/yelmo) over long simulations. It runs on CPU and GPU
(see the [performance page](https://TakisAngelides.github.io/Shakti.jl/dev/performance/)), and
includes optional extensions of the original model from GlaDS-2 and SUHMO (unfilled cavities, ice uplift,
channel-wall diffusion).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/TakisAngelides/Shakti.jl")
```

The backend (`"Threads"`, `"CUDA"`, `"Metal"`) and float type are set once, before `using Shakti`:

```julia
using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)
```

## Quick start

```julia
using Shakti

# A 4 km x 8 km synthetic glacier on a 40 x 80 grid, draining at x = 0.
nx, ny = 40, 80
grid = Grid(nx, ny, 4000.0, 8000.0)

mask = fill(GROUNDED, nx, ny)       # grounded ice, solved
mask[1, :] .= LAND                  # outlet: water leaves at atmospheric pressure
mask[end, :] .= OTHER_BASIN         # the other edges are closed
mask[:, [1, end]] .= OTHER_BASIN

zb     = zeros(nx, ny)                                         # bed elevation [m]
zs     = [sqrt(550.0^2 + (700.0^2 - 550.0^2) * x / 4000.0) for x in grid.x, y in grid.y] # surface [m]
b      = fill(0.01, nx, ny)                                    # initial gap height [m]
A_visc = fill(5e-25, nx, ny)                                   # ice rate factor [Pa^-3 s^-1]
G      = fill(0.05, nx, ny)                                    # geothermal flux [W/m2]
ub_x   = fill(1e-6, nx + 1, ny)                                # sliding velocity on x-faces [m/s]
ub_y   = zeros(nx, ny + 1)                                     # sliding velocity on y-faces [m/s]
ieb    = fill(1.0 / (365 * 86400.0), nx, ny)                   # water input from above [m/s] (1 m/a)
taub_x, taub_y = zeros(nx + 1, ny), zeros(nx, ny + 1)          # basal shear stress (recomputed by the sliding law)

p  = ModelParameters()
sl = RegularizedCoulombSlidingLaw(0.25)
state = State(grid)
set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

# 10 days of 1 h steps, head solved by Picard iteration with a sparse Cholesky solver.
ps  = PicardSolver(200, 1e-6, CholeskyDirectSolver(grid), grid)
sim = Simulation(grid, state, 240, floattype(3600.0), p, "fully_implicit", ["h", "N", "b"],
                 ConstantMeltInput(), sl; ps = ps,
                 which_observer = "IO", which_file_writer = "NetCDF",
                 tracked_times = 0:24:240, path = "output.nc")
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
