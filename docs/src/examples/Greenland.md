# [Greenland](@id Greenland)

The Greenland Ice Sheet from Yelmo restart files at 16 km and 8 km grid resolution. 
The default sheet is simulated for one year of 1-day steps. The loader is `load_yelmo` in
`docs/make_figures.jl`, which also makes the figures; the restart paths are local
(`docs/figure_paths.jl`).

```julia
using Shakti

# Load the data
d = load_yelmo(restart)
# d.xc, d.yc     cell-centre coordinates (m)
# d.DX           grid spacing (m)
# d.mask         GROUNDED / LAND / OCEAN cell types
# d.H, d.zb      ice thickness and bed elevation (m)
# d.A            flow-law parameter (Pa^-3 s^-1)
# d.G            geothermal heat flux (W m^-2)
# d.ub_x, d.ub_y sliding speed (m s^-1)

# Grid
Nx, Ny = size(d.H)
grid = Grid(Nx, Ny, (Nx - 1) * d.DX, (Ny - 1) * d.DX)

# Sliding law
sl = RegularizedCoulombSlidingLaw(0.25)   # regularized Coulomb law with exponent 0.25

# Model parameters
p = ModelParameters()                     # defaults

# Initial state: a gap of b0 everywhere, the surface above the gap, no surface input
b0 = 1e-3                                 # initial gap height (m)
b  = fill(b0, Nx, Ny)
zs = d.zb .+ d.H .+ b                     # ice surface elevation (m)
state = State(grid)
set_initial_conditions!(state, grid, p, sl, d.mask, d.A, d.zb, zs, b, d.G, d.ub_x, d.ub_y,
                        zeros(Nx, Ny), zeros(Nx + 1, Ny), zeros(Nx, Ny + 1))

# Solver
max_iter = 500                            # Picard iterations per step
tol = 1e-6                                # Picard tolerance
ps = PicardSolver(max_iter, tol, CUDSSDirectSolver(grid; structure = :general), grid)   # LUDirectSolver(grid) on CPU

# Run
nsteps = 365                              # 1 year
dt = 86400.0                              # 1 day (s)
output = ["N", "b", "Re", "po"]           # fields written to file
observer = IOObserver(output, [0, nsteps], NetCDFFileWriter(), "greenland.nc")
sim = Simulation(grid, state, nsteps, floattype(dt), p, ConstantMeltInput(), sl; ps = ps, observer = observer)
run!(sim)
```

| Grid | Cells | Grounded | Steps | `N/p_o` | Mean `N` | Max `b` |
|---|---|---|---|---|---|---|
| 16 km | 106 x 181 | 6 612 | 365 x 1 day | 0 to 1 | 1.93 MPa | 0.30 m |
| 8 km | 211 x 361 | 27 601 | 365 x 1 day | 0 to 1 | 1.86 MPa | 0.20 m |

Wall times are on the [Performance](@ref) page.

## 16 km

![Greenland 16 km, day 365](../figures/GrIS_16km.png)

## 8 km

![Greenland 8 km, day 365](../figures/GrIS_8km.png)

In the interior the water pressure is close to the overburden; towards the margins, where the ice
thins and the water drains to `LAND` at atmospheric pressure, `N/p_o` rises to 1. The sheet is
laminar under the divides and turbulent in the outlet systems.
