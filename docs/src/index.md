# Shakti.jl

A Julia solver for the SHAKTI subglacial hydrology model ([Sommers, Rajaram & Morlighem,
2018](https://gmd.copernicus.org/articles/11/2955/2018/)): a continuum model of hydraulic head, effective
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

See the [`API`](@ref) page for the full reference, or the
[README](https://github.com/TakisAngelides/Shakti.jl#readme) for installation and a quick-start
example.

## Examples

Three complete, runnable examples:

- [Seasonal melt input](@ref SeasonalMeltInput) -- a synthetic slab reproducing (a small, fast
  version of) the original SHAKTI paper's seasonal-cycle experiment.
- [Mask variants](@ref MaskVariants) -- three synthetic domain geometries (simple, barrier,
  semi-circle), showing what each mask value does to the drainage pattern.
- [Helheim Glacier](@ref Helheim) -- real data: rasterizing an unstructured ISSM mesh onto a
  regular grid and solving for the winter subglacial hydrology of a real Greenland outlet
  glacier.

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
| [`AbstractLinearSystem`](@ref) | How the linearized elliptic equation is represented | [`SparseAssembledLinearSystem`](@ref), [`MatrixFreeLinearSystem`](@ref) |
| [`AbstractLinearSolver`](@ref) | How that system is solved each Picard iteration | [`CholeskyDirectSolver`](@ref) (direct), [`CGIterativeSolver`](@ref) (iterative, with [`AMGPreconditioner`](@ref)/[`ChebyshevPreconditioner`](@ref)) |
| [`AbstractHeadScheme`](@ref) / [`AbstractGapScheme`](@ref) | Time-integration scheme for head / gap height | [`EllipticHeadScheme`](@ref) (Picard); [`ExplicitGapScheme`](@ref), [`ImplicitGapScheme`](@ref) |
| [`AbstractObserver`](@ref) | How output is recorded | [`NoObserver`](@ref), [`LiveObserver`](@ref) (in-memory), [`IOObserver`](@ref) (to disk: NetCDF/HDF5/JLD2/CSV) |
| [`Simulation`](@ref) | Bundles everything above and drives the time loop | [`Simulation`](@ref), [`run!`](@ref) |
