# API

## Backend selection

```@docs
Shakti.backend
Shakti.floattype
```

## Grid

```@docs
Grid
```

## State

```@docs
State
```

## Model parameters

```@docs
ModelParameters
pow
canonical_exponent
```

## Mask

```@docs
GROUNDED
OCEAN
LAND
OTHER_BASIN
compute_face_masks!
apply_mask_to_sliding!
```

## Melt input

```@docs
AbstractMeltInput
ConstantMeltInput
SeasonalMeltInput
update_ieb!
```

## K-face scheme

```@docs
AbstractKFaceScheme
Arithmetic
Harmonic
compute_K_face
```

## Sliding laws and melt rate

```@docs
AbstractSensibleHeatScheme
WithSensibleHeat
NoSensibleHeat
AbstractSlidingLaw
RegularizedCoulombSlidingLaw
PrescribedSlidingLaw
LinearSlidingLaw
initialize_taub!
compute_taub_x!
compute_taub_y!
compute_taub_xy!
compute_shear!
compute_potential!
compute_sensible!
compute_mdot!
```

## Linear solvers

```@docs
AbstractLinearSolver
AbstractDirectSolver
AbstractIterativeSolver
AbstractLinearSystem
SparseAssembledLinearSystem
MatrixFreeLinearSystem
CholeskyDirectSolver
CGIterativeSolver
solve_elliptic_linear_system!
solve_parabolic_linear_system!
```

## Preconditioners

```@docs
ChebyshevPreconditioner
update_chebyshev_bounds!
estimate_eigenvalue_bounds
AMGPreconditioner
update_amg!
```

## Elliptic (Picard) solver

```@docs
AbstractHeadRelaxation
NoHeadRelaxation
UnderHeadRelaxation
relax_h!
PicardSolver
elliptic_solver!
Picard_loop!
Picard_iteration!
```

## Parabolic solver

```@docs
parabolic_solver!
```

## Simulation

```@docs
AbstractHeadScheme
ParabolicHeadScheme
EllipticHeadScheme
AbstractGapScheme
ExplicitGapScheme
ImplicitGapScheme
Simulation
```

## Derived fields

```@docs
compute_H!
compute_po!
compute_h!
compute_abs_ub!
compute_pw!
compute_N!
compute_dhdx!
compute_dhdy!
compute_dhdxy!
compute_dpwdx!
compute_dpwdy!
compute_dpwdxy!
```

## Water flux, Reynolds number, transmissivity

```@docs
compute_q_x!
compute_q_y!
compute_q_xy!
compute_Re_x!
compute_Re_y!
compute_Re_xy!
compute_Re!
compute_K!
compute_q_and_Re_x!
compute_q_and_Re_y!
compute_q_and_Re_xy!
```

## Gap height

```@docs
AbstractOpenBySlidingScheme
WithOpenBySliding
NoOpenBySliding
compute_beta!
compute_b_x!
compute_b_y!
compute_b!
```

## Initial conditions

```@docs
set_initial_conditions!
```

## Checkpoint / restart

```@docs
save_checkpoint
load_checkpoint!
```

## Running a simulation

```@docs
run!
step!
step_h!
step_b!
```

## Observers (output recording)

```@docs
AbstractFileWriter
NetCDFFileWriter
HDF5FileWriter
JLD2FileWriter
CSVFileWriter
AbstractObserver
NoObserver
IOObserver
LiveObserver
get_observable
prepare!
observe!
openfile!
write2file!
finalize!
resume!
reopenfile!
```

## Animation

```@docs
make_mp4_mid
make_mp4_2d
get_moulin_ij
```
