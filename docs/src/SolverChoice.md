# Choosing a linear solver

Every Picard iteration solves one symmetric positive-definite system for the head. Which solver
is fastest depends mostly on **what kind of domain** you run, not on grid size alone. The
recommendation below comes from a benchmark on every real dataset in the test suite (2026-09-30:
30 real timesteps each, 8 CPU threads or one A100, head extrapolation order 1). Every solver
listed reproduces the Cholesky solution to at least 2e-5 relative in `N`.

## Recommendation

| Domain type | GPU available | CPU only |
|---|---|---|
| **Mountain / outlet glaciers** (Helheim, Drang Drung: steep beds, narrow channels, large conductivity contrasts) | `CUDSSDirectSolver(grid)` | `CholeskyDirectSolver(grid)`, or `CGIterativeSolver(grid, SparseAssembledLinearSystem; amg_refresh_every = 5)` at the finest grids |
| **Ice sheets** (Greenland, Pan-Antarctica, Thwaites: coarse cells, smoother coefficients) | `CGIterativeSolver(grid, MatrixFreeLinearSystem)` on the `"CUDA"` backend | `CGIterativeSolver(grid, SparseAssembledLinearSystem; amg = false, chebyshev_degree = 4)` |

On glacier grids CG needs many iterations (the operator is badly conditioned), so a direct
factorization wins; CUDSS is 7-12x faster than CPU Cholesky there. On ice-sheet grids Jacobi- or
Chebyshev-preconditioned CG converges in few iterations: matrix-free CG on a GPU is 13-58x faster
than CPU Cholesky, and Chebyshev CG is 1.7-2.5x faster than Cholesky on CPU.

## Measured time per step (median ms)

| Dataset | CPU Cholesky | CPU AMG (refresh 5) | CPU sparse Chebyshev | GPU CUDSS | GPU matrix-free Jacobi |
|---|---|---|---|---|---|
| Helheim 1 km (67x52) | 7.4 | 5.8 | 6.2 | 3.4 | 24 |
| Helheim 500 m (134x104) | 50 | 47 | 73 | **7.6** | 51 |
| Helheim 200 m (335x259) | 335 | 253 | 913 | **35** | 157 |
| Drang Drung 100 m | 118 | 138 | 470 | **15** | 246 |
| Drang Drung 50 m | 543 | 475 | 3121 | **49** | 543 |
| Drang Drung 25 m (407x799) | 2252 | 1943 | 26332 | **171** | 1234 |
| Greenland 16 km (106x181) | 62 | 45 | 26 | 9.5 | **3.1** |
| Pan-Antarctica 32 km (191x191) | 113 | 58 | 45 | 11 | **2.1** |
| Pan-Antarctica 16 km (381x381) | 870 | 395 | 344 | 92 | **7.4** |
| Thwaites 2 km | 1423 | 981 | 846 | 138 | **82** |

CPU timings come from jobs sharing nodes; treat differences under ~20% as ties.

## Things to avoid

- `CholeskyDirectSolver(grid; ordering = :metis)`: slower than the default AMD ordering on every
  dataset.
- AMG rebuilt every solve (`amg_refresh_every = 1`, the default): `amg_refresh_every = 5` was
  faster on every dataset.
- `ChebyshevPreconditioner(...; bounds = :lanczos)`: the Lanczos eigenvalue estimate can
  underestimate the largest eigenvalue, which makes the preconditioner indefinite and CG stall
  (final `N` wrong by up to 59% on glacier grids). The default `bounds = :gershgorin` is
  guaranteed safe. `CGIterativeSolver` now warns if CG stops before converging.
- `NewtonJFNKSolver`: never the fastest, and its converged state still differs from Picard's
  (1-65% in `N` across the datasets). Treat it as experimental.

## Independent of the solver

`Simulation(...; head_extrapolation_order = 1)` (the default) starts each head solve from a
linear extrapolation of the last two converged heads: 13% fewer Picard iterations overall and
11-44% less time per step, with the same answer. Order 2 saves a little more (18% fewer
iterations) when the forcing is smooth; order 3 saves nothing further.
