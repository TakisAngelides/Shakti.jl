# Shakti.jl

A Julia solver for the SHAKTI subglacial hydrology model ([Sommers, Rajaram & Morlighem,
2018](https://gmd.copernicus.org/articles/11/2955/2018/)): a continuum model of hydraulic head, effective
pressure, water flux, and drainage-system geometry (gap height) beneath an ice sheet, that
transitions smoothly between laminar (distributed) and turbulent (channelized) flow regimes
rather than treating them as separate model components.

Shakti solves for hydraulic head each timestep either via Picard iteration on the full nonlinear
elliptic equation ([`EllipticHeadScheme`](@ref), the default), or -- when a nonzero englacial
storage void ratio (`e_v`) is set -- via a single backward-Euler linear solve under the parabolic
head scheme ([`ParabolicHeadScheme`](@ref)). Either way, the gap height then evolves explicitly or
implicitly, using either a direct (sparse Cholesky) or iterative (preconditioned conjugate
gradient, with Chebyshev or algebraic-multigrid preconditioning) linear solver. It runs on CPU
(`Threads`) or GPU (`CUDA`/`Metal`) backends via
[ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl), selected once at load time
through `Preferences`-backed `backend`/`floattype` constants rather than a runtime argument, so
every kernel compiles for the right array/element type.

See the [`API`](@ref) page for the full reference, or the
[README](https://github.com/TakisAngelides/Shakti.jl#readme) for installation and a quick-start
example.

## Examples

Five complete, runnable examples, plus one static real-data writeup:

- [Seasonal melt input](@ref SeasonalMeltInput) -- a synthetic slab reproducing (a small, fast
  version of) the original SHAKTI paper's seasonal-cycle experiment.
- [Mask variants](@ref MaskVariants) -- three synthetic domain geometries (simple, barrier,
  semi-circle), showing what each mask value does to the drainage pattern.
- [Parabolic head scheme](@ref ParabolicScheme) -- runs the same synthetic slab under both the
  elliptic and parabolic head schemes from an identical spun-up state, and compares how far apart
  their `h` fields stay as both relax toward the same steady state.
- [All options](@ref AllOptions) -- a reference listing of every user-facing choice in Shakti
  (backends, solvers, sliding laws, melt inputs, schemes, observers, ...), gathered in one place
  and grouped by the source file that defines each choice.
- [Helheim Glacier](@ref Helheim) -- real data: rasterizing an unstructured ISSM mesh onto a
  regular grid and solving for the winter subglacial hydrology of a real Greenland outlet
  glacier.
- [Drang Drung Glacier](@ref DrangDrung) -- the same real-data pipeline applied to a small
  Himalayan alpine glacier; a static writeup with a figure from a real run rather than a
  live example (the finest-resolution run takes too long for a doc build).

## Package structure

The package is organized around a handful of core types, each with an abstract supertype where
there's more than one way to do that piece of the physics/numerics:

| Abstraction | Purpose | Concrete implementation(s) |
|---|---|---|
| [`Grid`](@ref) | Regular Cartesian grid geometry | [`Grid`](@ref) |
| [`State`](@ref) | Every hydrology field: hydraulic head, water/overburden/effective pressure, gap height, melt rate, water flux, Reynolds number, transmissivity, ... | [`State`](@ref) |
| [`ModelParameters`](@ref) | Physical constants (densities, viscosity, Glen's-law exponent, bed-bump geometry, ...) | [`ModelParameters`](@ref) |
| [`AbstractSlidingLaw`](@ref) | How basal shear stress `taub` (feeding frictional melt) is obtained | [`RegularizedCoulombSlidingLaw`](@ref), [`LinearSlidingLaw`](@ref), [`PrescribedSlidingLaw`](@ref) |
| [`AbstractMeltInput`](@ref) | Englacial-to-bed meltwater input (moulins/crevasses) | [`ConstantMeltInput`](@ref), [`SeasonalMeltInput`](@ref) |
| [`AbstractLinearSystem`](@ref) | How the linearized head equation is represented | [`SparseAssembledLinearSystem`](@ref), [`MatrixFreeLinearSystem`](@ref) |
| [`AbstractLinearSolver`](@ref) | How that system is solved each Picard iteration (or each parabolic timestep) | [`CholeskyDirectSolver`](@ref) (direct), [`CGIterativeSolver`](@ref) (iterative, with [`AMGPreconditioner`](@ref)/[`ChebyshevPreconditioner`](@ref)) |
| [`AbstractHeadScheme`](@ref) / [`AbstractGapScheme`](@ref) | Time-integration scheme for head / gap height | [`EllipticHeadScheme`](@ref) (Picard, `e_v == 0`); [`ParabolicHeadScheme`](@ref) (backward-Euler, `e_v != 0`); [`ExplicitGapScheme`](@ref), [`ImplicitGapScheme`](@ref) |
| [`AbstractObserver`](@ref) | How output is recorded | [`NoObserver`](@ref), [`LiveObserver`](@ref) (in-memory), [`IOObserver`](@ref) (to disk: NetCDF/HDF5/JLD2/CSV) |
| [`Simulation`](@ref) | Bundles everything above and drives the time loop | [`Simulation`](@ref), [`run!`](@ref) |
