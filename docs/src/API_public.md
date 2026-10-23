# API Reference

Every exported name, grouped the way a run uses them: grid and state, parameters and mask, the sheet
options, the solvers, the simulation and its output. The [Tutorial](@ref) explains when to pick
each option.

## Backend selection

```@docs
Shakti
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
FROZEN_BED
set_mask!
freeze_cells!
thaw_cells!
update_frozen_mask!
freeze_isolated!
freeze_on_capacity!
```

## Melt input

```@docs
AbstractMeltInput
ConstantMeltInput
SeasonalMeltInput
update_ieb!
GaussianMoulinMeltInput
combine_ieb!
set_ieb_external!
```

## K-face scheme

```@docs
AbstractKFaceScheme
Arithmetic
Harmonic
Upwind
ImplicitUpwind
compute_K_face
face_conductance
```

## Unfilled cavities, water-surface potential, ice uplift (GlaDS-2 sheet)

```@docs
AbstractCavityFilling
FilledCavities
UnfilledCavities
compute_b_w!
compute_b_w_iterate!
set_cavity_storage!
update_b_empty!
update_b_empty_budget!
clamp_b_empty!
AbstractHydraulicPotential
BedPotential
SheetTopPotential
update_gap_jumps!
AbstractIceUplift
set_uplift_storage!
update_b_uplift!
NoUplift
IceUplift
AbstractGapCoupling
ExplicitGapCoupling
ConsistentGapCoupling
```

## Sliding laws and melt rate

```@docs
MeltTerms
AbstractSlidingLaw
RegularizedCoulombSlidingLaw
PrescribedSlidingLaw
LinearSlidingLaw
RegularizedCoulombV0SlidingLaw
RegularizedCoulombFieldSlidingLaw
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
LUDirectSolver
CUDSSDirectSolver
CGIterativeSolver
solve_elliptic_linear_system!
solve_parabolic_linear_system!
```

## Preconditioners

```@docs
ChebyshevPreconditioner
update_chebyshev_bounds!
estimate_eigenvalue_bounds!
AMGPreconditioner
update_amg!
```

## Elliptic (Picard) solver

```@docs
AbstractHeadRelaxation
NoHeadRelaxation
UnderHeadRelaxation
relax_h!
AndersonHeadRelaxation
AbstractEllipticSolver
PicardSolver
elliptic_solver!
Picard_loop!
Picard_iteration!
refresh_head_dependents!
refresh_state!
NewtonJFNKSolver
newton_status
```

## Parabolic solver

```@docs
ParabolicPicardSolver
parabolic_solver!
Parabolic_loop!
Parabolic_iteration!
```

## Simulation

```@docs
AbstractHeadScheme
ParabolicHeadScheme
EllipticHeadScheme
AbstractGapScheme
ExplicitGapScheme
ImplicitGapScheme
FullyImplicitGapScheme
Simulation
AbstractHeadExtrapolation
NoHeadExtrapolation
HeadExtrapolation
reset_head_history!
AbstractTimeStepScheme
FixedTimeStep
AdaptiveTimeStep
target_time
```

## Derived fields

```@docs
compute_H!
compute_po!
compute_h!
compute_abs_ub!
compute_pw!
compute_N!
apply_boundary_pw!
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
compute_face_flux!
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
AbstractCreepLengthScheme
StandardCreep
CreepCutoff
compute_lc!
AbstractCellGapClamping
NoCellGapClamping
CellGapClamping
apply_cell_gap_clamping!
AbstractCellNClamping
NoCellNClamping
CellNClamping
SmoothCellNClamping
apply_cell_N_clamping!
```

## SUHMO channel-wall diffusion

```@docs
AbstractDiffusionScheme
NoDiffusion
WithDiffusion
compute_D!
solve_b_diffusion!
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
