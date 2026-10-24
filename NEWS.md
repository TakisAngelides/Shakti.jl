# Shakti.jl release notes

## 0.2.0

This release follows the 2026-10-23 code review. It breaks the API and changes several defaults; old drivers need
the migration below.

### Breaking: removed

- `NewtonJFNKSolver`. Use `PicardSolver`.
- `SmoothCellNClamping`, `CellGapClamping` and `clamp_budget`.
- `State` fields `b_empty`, `b_uplift`, `b_dry` and `w_old`, and the functions that updated them. The water `W` is
  the only prognostic water field.
- The lagged flux path and the standalone heat kernels.
- The 7-argument `fully_implicit_creep_update`.
- `limit_regime_crossing!` and `limit_filled_step!`: both are now `limit_head_step!`.
- Parabolic API names: `pps` is now `ps`, `sim.hs.pps` is now `sim.hs.ps`. One Picard loop serves the elliptic and
  parabolic head equations; direct `Picard_loop!` callers pass a `guard`.

### Breaking: changed defaults

- `omega = 1e-3` (Sommers et al. 2018, Table 2).
- `CUDSSDirectSolver` uses `structure = :general`.
- Gap coupling defaults to `ConsistentGapCoupling` (`ExplicitGapCoupling` remains for ISSM reproduction).
- The gap scheme is a keyword, `gap_scheme`, and defaults to `FullyImplicitGapScheme`.
- The time step's `safety_factor` is an extra cap on dt on top of the Picard-convergence control.
- `CGIterativeSolver(MatrixFreeLinearSystem)` is preconditioned by `AggregationMultigrid`.

### Deprecated (removed in 0.3)

- `ImplicitGapScheme`: use `FullyImplicitGapScheme`.
- The positional `Simulation` forms: the gap scheme as a string before `tracked_obs`, and `tracked_obs` with
  `which_observer`, `which_file_writer`, `tracked_times` and `path`. Pass `gap_scheme` and `observer` as keywords.
- `compute_q_and_Re_xy!` (use `compute_face_flux!`) and `compute_K!`.

### New

- `AdaptiveTimeStep`: dt controlled by Picard convergence.
- `BiCGStabIterativeSolver` and `AggregationMultigrid` for the default sheet with fields on the GPU.
- `LUDirectSolver(lagged = true)`: GMRES preconditioned by the last LU factorization.
- `WaterBudget` and `water_created(sim)`: the water created per step beyond sources and fluxes, for every sheet.
- Sheet presets `GlaDS2Sheet` and `ShaktiSheet` with keyword overrides; typed face schemes and observer objects.
- `OPTION_RULES`: one table of option combinations that are refused or warned about.
- `FilledCavities` warns once when a full gap leaves its regime (cells at `p_w < 0` or water created).
- `step!` warns when the head solve ends unconverged.

### Fixed

- The filled sheet's dry floor conducts by its water; the Drang Drung collapse under `ConsistentGapCoupling` at
  `b_min = 1e-3` is gone.
- One creep length at `N < 0` for every gap scheme.
- Sliding laws give no basal shear where `N < 0`.
- Wall diffusion uses one `D` per cell, averaged onto the faces (SUHMO Eq. 10).
- Mask changes report the water and seed reactivated cells dry.
- `|u_b|` is refreshed every step.
- Ice thickness is floored at zero and kept above the uplift layer.
- `LiveObserver` records each frame's model time.
