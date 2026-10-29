# Further numerical details

[Discretization](@ref) gives the discrete equations that a step solves. This page covers what surrounds
them: the safeguards that let the Picard iteration converge at long time steps, the handling of cells
that exchange no water, the time-step control, changes of the mask, and the linear solvers. None of it
changes the solution of a converged step; it decides whether, and how fast, the step converges.

## Upwind donors

Under `"implicit_upwind"` each face takes the conductance of its donor, the cell with the higher
``\hat{h}``. Near a flow divide the head difference across a face is small, and the donor can switch at
every iterate, which changes the matrix every time and can keep the iteration from settling. So the
donors are tracked through the step:

1. At the start of a step every face is free. A face takes the higher-``\hat{h}`` cell as its donor at
   its first iterate with flow, and switches each time the flow reverses.
2. After four reversals within the step the face keeps its donor for the rest of the step.
3. Once the head has converged, a face that was frozen on its downstream side (the flow now runs into
   the donor) is moved to its upstream side, and the iteration resumes. A face that reverses again after
   that takes the conductance of its thinner cell. This polishing runs at most three times per step.

The water-depth Newton terms ([Picard iteration](@ref)) follow the donor in use, so the matrix stays
consistent with the conductance it was built from.

The donor's water depth is its water at the end of the step at the current head iterate. Its melt
opening is the one at the start of the step, unless the melt at the iterate gives less water; this keeps
the feedback ``W \to K \to q \to \dot m \to W`` of a growing channel out of the iteration, while the head
rows themselves use the melt at the iterate. On the filled sheet's dry floor the depth is the water the
gap update leaves, not the gap held at ``b_\min``, so a drained cell does not conduct. Under
`WithDiffusion` it is the smaller of the start-of-step gap and the gap of the last gap solve.

## Limiting the head iterate

Each head row has kinks where its slope jumps: the bed (unfilled to filled), the overburden head (filled
to uplifted) and the boundary head of an outflow-only face. A Newton step taken from one side of a kink
can land far into the other, and the next step lands far back, so the iteration cycles. After each linear
solve the new head is limited cell by cell, in this order:

- **Filled cap.** Under `ConsistentGapCoupling` the step of a filled grounded cell is capped at
  ``|\Delta N| \le \max(|N|/2,\, p_o/4)``, and at least 1 m of head. On thick ice the gap update closes a
  thin gap within one step for most ``N``, so near ``N = p_o`` it is nearly flat in ``N`` and the Newton
  step overshoots by kilometres. On the filled sheet without uplift, where ``N`` may go negative, ``N`` may
  fall at most ``p_o/4`` below ``\min(N^k, 0)`` per iterate, so it cannot run away geometrically. The cap is
  off on the filled sheet with englacial storage (``e_v > 0``), whose rows stay steep in ``N``.
- **Regime stops.** An update that crosses the bed or the overburden head stops just inside the new regime
  (by a millionth of the gap), so the next solve uses the row of the regime the head is in.
- **Limits inside a regime.** An uplift layer at most doubles per iterate, and the head of an unfilled cell
  sinks below the bed at most twice as deep, or by three quarters of the gap if that is more.
- **Outflow stops.** An update that crosses the boundary head of an outflow-only face stops at it, so the
  face is in the state it ends in for the next solve.
- **Oscillation damping.** An update that reverses the previous one by more than half of it is cut to half,
  which brackets a root at a kink and halves the bracket each iterate, as bisection does.

A converged head is a fixed point of the limited iteration, so the limits do not change the solution.
With `UnderHeadRelaxation(alpha)` the limited head is further blended with the previous one,
``h^{k+1} \leftarrow \alpha h^{k+1} + (1-\alpha) h^k``; `AndersonHeadRelaxation` accelerates it instead.

## The first iterate

The iteration starts from the head extrapolated in time through the last converged heads
(`head_extrapolation_order`, 1 = linear by default), with Lagrange weights on the actual step end times so
that a varying ``\Delta t`` is handled exactly. Only converged heads enter the history: the first
extrapolation happens at the third step, and a step that does not converge clears the history. Only
grounded cells are extrapolated. The extrapolated head is then limited as above, measured from the last
converged head and without the filled cap. A cell that changed regime in the last step moved by tens of
metres, and extrapolating that move into the uplifted regime would start the step with a blister tens of
metres deep.

## Closed cells

A grounded cell whose four faces are all closed (frozen or `OTHER_BASIN` neighbours, closed outflow
faces, the domain edge) exchanges no water. With ``e_v = 0`` its row has no off-diagonal entries, and its
diagonal, the slope of its stored water, is zero at ``N = 0`` and tiny near it, where a Newton step
overshoots by orders of magnitude. Its head is therefore solved exactly. The water at the end of the step
is known, ``T = W^n + \Delta t\,(\dot m/\rho_w + i_{eb})``, less the wall diffusion under `WithDiffusion`.

- Under `ConsistentGapCoupling`, if uplift is allowed and ``T > \mathcal{G}(0)``, the ice is lifted:
  ``h = h_o + T - \mathcal{G}(0)``. If unfilled cavities are allowed and ``T < \mathcal{G}(p_o)``, the gap is
  partly empty: ``h = z_b - (\mathcal{G}(p_o) - T)``. Otherwise ``\mathcal{G}(N) = T`` is solved for ``N`` by
  60 bisections, on ``[0, p_o]`` on the GlaDS-2 sheet, and on ``[-10 p_o, 10 p_o]`` on the filled sheet
  (``1000\, p_o`` on its dry floor), within ``[N_\min, N_\max]``.
- Under `ExplicitGapCoupling` the creep must close the gap as fast as sliding and melt open it,
  ``A |N|^{n-1} N\, l_c = R`` with ``R = \beta |\mathbf{u}_b| - \dot m (1/\rho_w - 1/\rho_i) - i_{eb}`` (plus
  ``(b^n - W^n)/\Delta t`` on the GlaDS-2 sheet), so ``N = \operatorname{sign}(R)\,(|R| / (A l_c))^{1/n}``,
  clamped to ``[N_\min, N_\max]``. Without creep (``A l_c = 0``) the gap cannot close, and ``N = p_o`` if
  ``R > 0``, else 0. The unfilled and uplifted cases are taken first, as above: the gap is partly empty if
  even the closure at ``p_o`` cannot balance ``R``, and the ice is lifted if ``R < 0``.

With ``e_v > 0`` the englacial storage keeps every diagonal positive, and closed cells take the ordinary
row. A patch of several cells closed as a whole, all at ``N = 0``, has no head level of its own and is not
covered.

## Time-step control

Under `FixedTimeStep` every step has the same ``\Delta t``. A step whose head solve does not converge is
taken as it is, counted in `water_created(sim).unconverged`, and reported once with the water it created.

`AdaptiveTimeStep(grid, state; dt_min, dt_max, target_time, ...)` changes ``\Delta t`` from step to step:

- **Rejection.** A step whose head solve does not converge, at ``\Delta t > \Delta t_\min``, is retried
  from its start at half the step (no less than ``\Delta t_\min``). Only the head needs resetting: the gap,
  the water and the head history change only once a step is accepted.
- **Growth.** After `grow_after` (3) accepted steps in a row that each converged within `fast_iters` (30)
  Picard iterations, the step grows by `grow` (1.5), up to ``\Delta t_\max``.
- **Rate cap.** With `safety_factor` set, ``\Delta t \le \text{safety\_factor} / r``, where ``r`` is the
  largest (or, with `use_percentile = true`, the `percentile` quantile) of the gap's relaxation rate
  ``C\, l_c'(b) + \gamma`` over the grounded cells above ``b_\min``.
- **Landing on the target.** The last step is cut to end exactly at `target_time`, and the one before it
  is shortened when the remainder would otherwise be less than ``\Delta t_\min``.

## Freeze-on capacity

With `limit_freeze_on` the melt rate is floored at minus the freeze-on that the next gap update can absorb
without the gap falling below ``b_\min`` ([Melt on the grid](@ref)). Each term is evaluated where the gap
scheme evaluates it: opening by sliding and creep at ``b_\min`` for `FullyImplicitGapScheme`, at ``b^n`` for
`ExplicitGapScheme`, and for `WithDiffusion` from the local terms of its gap system, ignoring the diffusion
(which only moves gap volume between cells). When a host ice model asks for the capacity over its own,
longer step (`freeze_on_capacity!(C, sim; dt)`), the room in the gap is spread over that step, and creep and
sliding are evaluated at the current gap, since over many Shakti steps they act on the actual gap, not on
``b_\min``. Under `UnfilledCavities` and on the dry floor the capacity is also capped by the water present,
``(\rho_w/\rho_i)\, W / \Delta t``.

## Mask changes

A coupled ice model can change which cells are solved, between steps:

- `freeze_cells!` turns grounded cells into `FROZEN_BED`: ``b = W = 0``, ``p_w = 0``, ``h = z_b``, and so
  ``N = p_o``. The water they held is discarded, not refrozen.
- `thaw_cells!` turns `FROZEN_BED` cells back into grounded ones, at ``b = b_\min`` with no water (an
  empty cavity, or a dry floor), keeping their head as the start of the next solve.
- `update_frozen_mask!` freezes grounded cells whose basal temperature relative to the pressure melting
  point is below `T_freeze` and thaws frozen cells at or above `T_freeze + T_hysteresis`.
- `set_mask!` takes the host's mask, keeping frozen cells frozen. A cell that leaves the grounded set loses
  its gap and water and takes the boundary pressure of its new category; a cell that joins it gets
  ``b = b_\min`` with no water, and its head starts from the boundary pressure it had.

Each refreshes the face masks, ``N``, ``\beta`` and ``l_c``, and reports the water it discarded.
`update_frozen_mask!` and `set_mask!` then freeze any grounded cell left with no neighbour that is grounded,
`LAND` or `OCEAN`, since its water has nowhere to go and its row could be singular. The `Simulation` methods
of these functions also bring the water within what the sheet allows, and refresh the fields that depend on
the simulation's options (cell ``N`` clamps, outflow-only faces).

## Linear solvers

The head matrix is a five-point stencil. It is symmetric and positive definite with every face scheme
except `"implicit_upwind"`, whose Newton terms make it non-symmetric (it keeps the sign pattern of an
M-matrix). The solvers:

| Solver | Matrix it solves | Representation | Fields on |
|:-------|:-----------------|:---------------|:----------|
| `LUDirectSolver` | any | assembled | CPU |
| `CholeskyDirectSolver` | symmetric | assembled | CPU |
| `CUDSSDirectSolver` | any (`structure = :general`, LU) or symmetric (`:spd`, Cholesky) | assembled on the CPU, factorized on an NVIDIA GPU | CPU |
| `CGIterativeSolver(grid, SparseAssembledLinearSystem)` | symmetric | assembled | CPU |
| `CGIterativeSolver(grid, MatrixFreeLinearSystem)` | symmetric | matrix-free | CPU or GPU |
| `BiCGStabIterativeSolver` | any | matrix-free | CPU or GPU |

An assembled solver builds the sparse matrix explicitly, which only the CPU can do, so its fields stay on the
CPU. A matrix-free solver never forms the matrix: it applies the five-point stencil from the per-cell
coefficients, so it runs wherever the fields live, and it is the only kind that runs with the fields on a GPU.

The assembled matrix's sparsity pattern is built once, so each iterate only rewrites its values. The
iterative solvers warm-start from the current head. A matrix-free solver is preconditioned by an
aggregation-multigrid V-cycle by default (or Jacobi; CG also takes Chebyshev), and CG on the assembled matrix
by algebraic multigrid. `LUDirectSolver(grid; lagged = true)` reuses its last
factorization as the preconditioner of a GMRES solve and refactorizes only when GMRES stalls. Under
`WithDiffusion` the gap system is symmetric positive definite, and takes its own solver (Cholesky, cuDSS
with `:spd`, or CG). See [the tutorial](@ref "8. Solvers") for when to use which.

## Diagnostic fields

Some fields are computed for output only and never read by the solve: the cell-centred conductance
``K = g b^3 / (12 \nu (1 + \omega \mathrm{Re}))``, the cell Reynolds number, the mean of its four faces, and
the face gaps `b_x`, `b_y`, the mean of the two cells (the nearest cell's gap on the domain edge). The solve
reads the face conductances `K_x`, `K_y` built from the water depth.
