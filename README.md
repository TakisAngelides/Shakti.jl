# Shakti

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://TakisAngelides.github.io/Shakti.jl/dev/)
[![](https://img.shields.io/badge/license-GNU_GPL_3.0-green.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![Build Status](https://github.com/TakisAngelides/Shakti.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TakisAngelides/Shakti.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia solver for the SHAKTI subglacial hydrology model ([Sommers and others,
2018](https://doi.org/10.3389/feart.2018.00104)): a continuum model of hydraulic head, effective
pressure, water flux, and drainage-system geometry (gap height) beneath an ice sheet, that
transitions smoothly between laminar (distributed) and turbulent (channelized) flow regimes
rather than treating them as separate model components.

Shakti solves the nonlinear elliptic equation for hydraulic head via Picard iteration each
timestep, then evolves the gap height explicitly or implicitly, using either a direct (sparse
Cholesky) or iterative (preconditioned conjugate gradient, with Chebyshev or algebraic-multigrid
preconditioning) linear solver. It runs on CPU (`Threads`) or GPU (`CUDA`/`Metal`) backends via
[ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl), selected once at load time
through `Preferences`-backed `backend`/`floattype` constants rather than a runtime argument, so
every kernel compiles for the right array/element type.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/TakisAngelides/Shakti.jl")
```

To select a backend other than the default (`Threads`, `Float64`), set the `backend`/`floattype`
preferences before `using Shakti` (see `Preferences.jl`):

```julia
using Preferences
set_preferences!("Shakti", "backend" => "CUDA", "floattype" => "Float32"; force = true)
```

## Quick start

```julia
using Shakti

# 1. Build a grid: nx*ny cells over a domain of size lx*ly (meters).
grid = Grid(nx, ny, lx, ly)

# 2. Physical constants, melt input, and the basal sliding law.
p  = ModelParameters()               # see its docstring for every keyword/default
mi = ConstantMeltInput()             # or SeasonalMeltInput(...) for a seasonal cycle
sl = RegularizedCoulombSlidingLaw(C) # or LinearSlidingLaw(grid, C) / PrescribedSlidingLaw()

# 3. Build the state and populate it from your input data: mask (GROUNDED/OCEAN/LAND/
#    OTHER_BASIN, see src/mask.jl), bed/surface elevation, ice thickness via gap height,
#    Glen's-law rate factor, geothermal flux, and basal sliding velocity.
state = State(grid)
set_initial_conditions!(state, grid, p, mi, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

# 4. Choose a linear solver for the Picard/elliptic head solve...
ls = CholeskyDirectSolver(grid)               # fastest at every grid size benchmarked so far, CPU-only
# ls = CGIterativeSolver(grid, MatrixFreeLinearSystem) # GPU-capable alternative
ps = PicardSolver(500, 1e-6, ls, grid)

# 5. ...and wrap everything in a Simulation.
sim = Simulation(grid, state, tsteps, dt, p, "implicit", ["h", "N", "b", "mdot"], mi, sl;
                  ps = ps, which_observer = "IO", which_file_writer = "NetCDF",
                  tracked_times = 0:tsteps, path = "output.nc")

# 6. Run. state.h/state.N/state.b/... hold the solution at every timestep in between,
#    and the tracked fields above are written to output.nc as the run progresses.
run!(sim)
```

See the [online documentation](https://TakisAngelides.github.io/Shakti.jl/dev/) for the full API
reference, and `test/runtests.jl` for complete, runnable setups (synthetic grids exercising every
sliding law, melt input, linear solver, and observer combination).

## Package structure

The package is organized around a handful of core types, each with an abstract supertype where
there's more than one way to do that piece of the physics/numerics:

| Abstraction | Purpose | Concrete implementation(s) |
|---|---|---|
| `Grid` | Regular Cartesian grid geometry | `Grid` |
| `State` | Every hydrology field: hydraulic head, water/overburden/effective pressure, gap height, melt rate, water flux, Reynolds number, transmissivity, ... | `State` |
| `ModelParameters` | Physical constants (densities, viscosity, Glen's-law exponent, bed-bump geometry, ...) | `ModelParameters` |
| `AbstractSlidingLaw` | How basal shear stress `taub` (feeding frictional melt) is obtained | `RegularizedCoulombSlidingLaw`, `LinearSlidingLaw`, `PrescribedSlidingLaw` |
| `AbstractMeltInput` | Englacial-to-bed meltwater input (moulins/crevasses) | `ConstantMeltInput`, `SeasonalMeltInput` |
| `AbstractLinearSystem` | How the linearized elliptic equation is represented | `SparseAssembledLinearSystem`, `MatrixFreeLinearSystem` |
| `AbstractLinearSolver` | How that system is solved each Picard iteration | `CholeskyDirectSolver` (direct), `CGIterativeSolver` (iterative, with `AMGPreconditioner`/`ChebyshevPreconditioner`) |
| `AbstractHeadScheme` / `AbstractGapScheme` | Time-integration scheme for head / gap height | `EllipticHeadScheme` (Picard); `ExplicitGapScheme`, `ImplicitGapScheme` |
| `AbstractObserver` | How output is recorded | `NoObserver`, `LiveObserver` (in-memory), `IOObserver` (to disk: NetCDF/HDF5/JLD2/CSV) |
| `Simulation` | Bundles everything above and drives the time loop | `Simulation`, `run!` |

### Source layout

- `Shakti.jl` -- module entrypoint: backend/floattype setup (`Preferences`-backed), file
  includes, and the public API's `export` list.
- `grid.jl` -- `Grid`.
- `state.jl` -- `State`.
- `model_parameters.jl` -- `ModelParameters`, plus fast-exponentiation helpers for
  Glen's-flow-law-style powers.
- `mask.jl` -- the `GROUNDED`/`OCEAN`/`LAND`/`OTHER_BASIN` mask convention and the face-validity
  bookkeeping derived from it.
- `melt_input.jl` -- `AbstractMeltInput` and its implementations.
- `k_face_scheme.jl` -- how hydraulic transmissivity is averaged across a cell face
  (`Arithmetic`/`Harmonic`).
- `melt_rate.jl` -- sliding laws (`AbstractSlidingLaw`) and melt-rate computation (geothermal +
  frictional + potential-energy + sensible-heat contributions).
- `field_gradients.jl` -- hydraulic-head and water-pressure gradients.
- `water_flux.jl` -- water flux, Reynolds number, and hydraulic transmissivity.
- `gap_height.jl` -- gap-height (`b`) evolution.
- `linear_solver.jl` -- the assembled-sparse and matrix-free linear-system representations, and
  their direct/iterative solvers.
- `preconditioner.jl` -- Chebyshev semi-iteration and algebraic-multigrid preconditioners for
  `CGIterativeSolver`.
- `elliptic_solver.jl` -- the Picard iteration that solves the nonlinear elliptic equation for
  head each timestep.
- `simulation.jl` -- `Simulation`.
- `initial_conditions.jl` -- `set_initial_conditions!`.
- `static_fields.jl`, `pressure.jl` -- fields derived directly from geometry/head (thickness,
  overburden pressure, water pressure, effective pressure).
- `run.jl` -- the time-stepping loop (`run!`/`step!`), plus checkpoint/restart wiring.
- `checkpoint.jl` -- save/restore a running simulation's full state.
- `observer.jl` -- output recording, to disk (NetCDF/HDF5/JLD2/CSV) or in memory.
- `animation.jl` -- turning a `LiveObserver`'s recorded history into `.mp4` animations.

## Testing

```julia
using Pkg
Pkg.test("Shakti")
```

The test suite exercises the full pipeline (initial conditions -> Picard solve -> time-stepping)
on synthetic grids: `RegularizedCoulombSlidingLaw`/`LinearSlidingLaw`, every linear
solver/preconditioner combination (`CholeskyDirectSolver`, `CGIterativeSolver` with Jacobi/
AMG/Chebyshev), every observer/file-writer combination, and checkpoint/restart. Not yet covered:
`PrescribedSlidingLaw` and `SeasonalMeltInput`.

## License

GNU General Public License v3.0 -- see [LICENSE](LICENSE).
