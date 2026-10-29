# Discretization

This page shows how the equations of [Model equations](@ref) are solved on the grid. Space is
discretized by finite volumes on a regular grid, and time by backward Euler. Each time step first solves
the head equation by a Picard iteration and then moves the gap. Unless said otherwise, the page describes
the defaults: the GlaDS-2 sheet, `ConsistentGapCoupling`, `"implicit_upwind"` faces and
`FullyImplicitGapScheme`. The safeguards that make the iteration robust, and the bookkeeping around the
solve, are in [Further numerical details](@ref).

## Grid

`Grid(nx, ny, lx, ly)` has ``n_x \times n_y`` cells with centres
``\mathbf{x}_{ij} = ((i-1)\Delta x, (j-1)\Delta y)``, where ``\Delta x = l_x / (n_x - 1)`` and
``\Delta y = l_y / (n_y - 1)``, so ``l_x`` and ``l_y`` are the distances between the outermost centres. The
fields ``b``, ``W``, ``h``, ``p_w``, ``N`` and ``\dot m`` live at the cell centres. The conductance ``K``,
the Reynolds number ``\mathrm{Re}`` and the flux ``q`` live on the faces between them (a staggered grid:
`q_x` has size ``(n_x + 1) \times n_y`` and `q_y` has size ``n_x \times (n_y + 1)``). The face formulas below
are written for the ``x`` face ``i - \tfrac12`` between cells ``i - 1`` and ``i``; the ``y`` faces are the
same with ``\Delta y``.

## Time step

A step ``n \to n+1`` of length ``\Delta t`` (`step!`) does four things:

1. **Prepare.** From the gap ``b^n`` and the water ``W^n`` at the start of the step it computes the
   overburden ``p_o``, the jumps of the gap across the faces (for ``\hat{h} = h + b``) and the freeze-on
   limit. These, and the opening coefficient ``\beta`` and creep length ``l_c`` left by the previous step,
   are held for the whole step. A coupled ice model may have changed ``z_s``, ``z_b`` and
   ``\mathbf{u}_b`` before this.
2. **Solve for the head.** The head equation is solved for ``h^{n+1}`` by Picard iteration. Every field that
   depends on the head (``p_w``, ``N``, ``\mathbf{q}``, ``\mathrm{Re}``, ``\boldsymbol{\tau}_b`` and
   ``\dot m``) is recomputed at every iterate, so at convergence all of them are evaluated at ``n+1``.
3. **Move the gap.** The gap scheme ``\mathcal{G}`` gives
   ``b^{n+1} = \mathcal{G}(b^n; N^{n+1}, \dot m^{n+1})``.
4. **Update the water.** It sets ``W^{n+1}`` and refreshes ``\beta`` and ``l_c`` from ``b^{n+1}`` for the
   next step.

The head equation contains ``\partial b / \partial t``, which step 3 has not computed yet, so step 2 must
assume a value for it. The gap coupling decides which. Under `ConsistentGapCoupling` the head rows
assume exactly ``\mathcal{G}``, so steps 2 and 3 store the same water and the water budget closes at any
``\Delta t``. Under `ExplicitGapCoupling` (Shakti's original scheme) they assume
``b^{n+1} - b^n = \Delta t\, \dot b(N^{n+1}, b^n)``. This differs from ``\mathcal{G}`` where creep is fast
(``\Delta t A |N|^n > 1``), and the difference is water created or lost
([Gap coupling](@ref GapCoupling)).

## Face flux

The flux across a face is driven by the difference of the hydraulic head ``\hat{h} = h + b`` between its
two cells, with the gap held at ``b^n``:

```math
(\nabla\hat{h})_{i-\frac12} = \frac{(h_i + b^n_i) - (h_{i-1} + b^n_{i-1})}{\Delta x} .
```

Under `BedPotential` the ``b^n`` terms are dropped. On a face to an `OCEAN` or `LAND` cell the boundary
side's gap counts as 0, since its fixed head is already the level of its water.

The laminar conductance of the face is

```math
K_{0,f} = \frac{g}{12\nu} \, \overline{W^3}_f ,
```

where ``\overline{W^3}_f`` combines the two cells' water depths as `k_face_choice` says:

| `k_face_choice` | ``\overline{W^3}_f`` |
|:----------------|:---------------------|
| `"implicit_upwind"` (default on the free-surface sheets) | the donor's ``W^3``, with ``W`` the donor's water at the end of the step |
| `"upwind"` | the donor's ``W^3``, with ``W`` the donor's water at the start of the step |
| `"arithmetic"` (default on the filled sheet) | ``(W_{i-1}^3 + W_i^3)/2`` |
| `"harmonic"` | ``2 W_{i-1}^3 W_i^3 / (W_{i-1}^3 + W_i^3)`` |
| `"gap_mean"` | ``\left((W_{i-1} + W_i)/2\right)^3``, the face rule of SUHMO |

The donor is the cell with the higher ``\hat{h}``, the one the water flows out of. Taking the donor's depth
keeps an empty cell from draining water it does not have, which is why the averaged faces are refused on
the free-surface sheets. Within a step, a face whose donor has reversed four times keeps its donor, so
the iteration cannot cycle on it. A face to an `OCEAN` or `LAND` cell takes the grounded cell's ``W``, and
a face to `OTHER_BASIN`, `FROZEN_BED` or the edge of the grid has ``K_f = 0``.

The boundary faces are outflow-only by default (`outflow_only_land`, `outflow_only_ocean`): a face to a
`LAND` or `OCEAN` cell is closed, ``K_f = 0``, while the boundary head lies above the grounded cell's water
level ``h + b``, so the boundary drains the ice but never feeds it. Whether each such face is open is decided
again from the head at every Picard iterate.

The turbulent correction follows from the flux law. With ``D_f = K_{0,f} |(\nabla\hat{h})_f| / \nu``, the
Reynolds number ``\mathrm{Re} = |q|/\nu`` solves ``\omega \mathrm{Re}^2 + \mathrm{Re} - D_f = 0``, so

```math
\mathrm{Re}_f = \frac{2 D_f}{1 + \sqrt{1 + 4 \omega D_f}}, \qquad
K_f = \frac{K_{0,f}}{1 + \omega \mathrm{Re}_f}, \qquad
q_f = -K_f (\nabla\hat{h})_f .
```

This form of the root avoids cancellation as ``\omega \to 0`` and gives ``|q_f| = \nu \mathrm{Re}_f``
exactly, with no lag between ``q`` and ``\mathrm{Re}``. The same ``K_f`` is assembled into the matrix and
used by every diagnostic (``\mathbf{q}``, ``\mathrm{Re}``, ``\dot m``, ``\mathcal{D}``), so the water that
the solve conserves and the water that melts the ice are the same.

## The discrete head equation

For a cell ``P`` with neighbours ``F`` across its four faces ``f``, the divergence of the face fluxes is

```math
(\nabla\cdot\mathbf{q})_P = \frac{q_{i+\frac12} - q_{i-\frac12}}{\Delta x} + \frac{q_{j+\frac12} - q_{j-\frac12}}{\Delta y} .
```

Integrating the mass balance over the cell and over the step gives the row of cell ``P``:

```math
\sum_f \frac{K_f}{\Delta_f^2} \left( h_P - h_F \right)
  + \frac{b_e(h_P) - b_e(h_P^n)}{\Delta t}
  = \frac{\dot m_P}{\rho_w} + i_{e\to b,P}
  - \frac{\mathcal{W}_P(h_P) - W^n_P}{\Delta t}
  - \sum_f \frac{K_f}{\Delta_f^2} \left( b^n_P - b^n_F \right) ,
```

where ``\Delta_f`` is ``\Delta x`` or ``\Delta y``. On the left are the net outflow and the englacial storage.
On the right are the sources, the change of the water the cell holds, and the part of the outflow driven by
the known gap jumps. ``\mathcal{W}_P(h)`` is the water the cell holds at the end of the step if its head is
``h``. By the closure of [Three regimes of the sheet](@ref),

```math
\mathcal{W}_P(h) =
\begin{cases}
\mathcal{G}_P(p_o) - (z_b - h) & h < z_b \quad \text{(unfilled)} \\
\mathcal{G}_P\!\left(p_o - \rho_w g (h - z_b)\right) & z_b \le h \le h_o \quad \text{(filled)} \\
\mathcal{G}_P(0) + (h - h_o) & h > h_o \quad \text{(uplifted)}
\end{cases}
```

with ``\mathcal{G}_P(N)`` the gap after the step at effective pressure ``N``. The three branches agree at
``h = z_b`` and ``h = h_o``, so a cell fills, pressurizes and lifts the ice continuously. In the unfilled
and uplifted branches ``\mathcal{W}_P`` grows with ``h`` with slope 1, which puts ``1/\Delta t`` on the
diagonal. In the filled branch the slope is ``\rho_w g\, |\partial \mathcal{G}_P / \partial N|`` (closure
grows with ``N``), computed by a central difference. On the filled sheet's dry floor the filled branch
moves the water ``W^n`` instead of the gap, with no lower clamp at ``b_\min``, so a drained cell keeps a
storage slope.

The englacial storage is ``b_e(h) = e_v (h - z_b)``. With unfilled cavities it stops at the bed, and with
uplift at the overburden head, since beyond them the water is in ``W`` instead; on the GlaDS-2 sheet
``b_e(h) = e_v \min(\max(h - z_b, 0),\, p_o/(\rho_w g))``. It is Newton-linearized at the iterate and adds
``e_v / \Delta t`` to the diagonal where its slope is not zero.

Under `ExplicitGapCoupling` the end-of-step water is replaced by the gap rate at ``b^n``. The term
``(\mathcal{W}_P - W^n_P)/\Delta t`` becomes ``(b^n_P - W^n_P)/\Delta t`` (zero in a full gap) plus

```math
\frac{\dot m_P}{\rho_i} + \beta(b^n_P)\, |\mathbf{u}_b|_P - A_P |N_P|^{n-1} N_P\, l_c(b^n_P) ,
```

with ``N_P = p_o - \rho_w g (h_P - z_b)`` in the filled branch, ``p_o`` in the unfilled branch and 0 in
the uplifted branch, where the slope-1 storage stays. In a filled cell this is the right-hand side of
[Head equation](@ref) on the grid, the scheme of [Sommers2018](@cite).

Some rows are special:

- A grounded cell whose four faces are all closed, with ``e_v = 0``, exchanges no water. Its water at the
  end of the step, ``W^n + \Delta t (\dot m / \rho_w + i_{e\to b})``, is known, and ``\mathcal{W}_P(h)`` equal
  to it is solved for ``h`` exactly: by bisection, or under `ExplicitGapCoupling` by inverting the creep
  law in closed form.
- An `OCEAN` or `LAND` neighbour's head is known, so its term moves to the right-hand side. This keeps the
  matrix symmetric wherever the face scheme allows it.
- `OCEAN` and `LAND` cells have the row ``h_P = `` their boundary head, and `OTHER_BASIN` and
  `FROZEN_BED` cells the row ``h_P = `` their current head.

From the head, every cell has
``p_w = \rho_w g (h - z_b)``, floored at 0 with unfilled cavities and capped at ``p_o`` with uplift, and
``N = \operatorname{clamp}(p_o - p_w,\, N_\min,\, N_\max)``, then clamped again cell by cell by
`CellNClamping`. The clamped ``N`` is the one the head rows and the gap update use; ``p_w`` is not clamped, so
where a clamp binds, ``N \ne p_o - p_w``. On the free-surface sheets ``N`` already lies in ``[0, p_o]``, and a clamp
tighter than ``[0, \infty)`` is refused, since it would contradict the unfilled (``N = p_o``) and uplifted
(``N = 0``) rows. By default there is no clamp.

## Picard iteration

The row is nonlinear in ``h``: through ``N`` in ``\mathcal{G}``, through ``W`` and ``\mathrm{Re}`` in
``K_f``, and through ``\mathbf{q}`` and ``\boldsymbol{\tau}_b(N)`` in ``\dot m``. An iterate
``k \to k+1`` (`PicardSolver`) does the following:

1. **Linearize.** Evaluate ``K_f``, ``\dot m`` and the regime of every cell at ``h^k``. Each row takes the
   branch of ``\mathcal{W}_P`` that ``h^k_P`` lies in and keeps it for the solve, even if ``h^{k+1}_P`` ends in
   another regime (step 4 keeps such crossings small). The branch is linearized about ``h^k_P``:
   ``\mathcal{W}_P(h) \approx \mathcal{W}_P(h^k) + \mathcal{W}_P'(h^k)\,(h - h^k)``.
   Under `ExplicitGapCoupling` the creep term is linearized the same way, which puts
   ``n \rho_w g A |N|^{n-1} l_c`` on the diagonal.
2. **Add the water-depth Newton terms.** Under `"implicit_upwind"`, ``K_f`` takes the donor's water at the
   end of the step, ``\mathcal{W}_d(h^k)``. Its melt opening is taken from the start of the step, or from
   the iterate where that gives less water, so that the channel feedback ``W \to K \to q \to \dot m \to W``
   does not run inside the iteration. The row also takes the Newton term of the face flux in the donor's head,
   ``\partial |q_f| / \partial h_d = 3 K_f |(\nabla\hat{h})_f|\, (1 + \omega\mathrm{Re}_f) / \left(W_d (1 + 2\omega\mathrm{Re}_f)\right) \cdot \mathcal{W}_d'``.
   The iteration is then Newton's method in the water depth, which converges at long ``\Delta t``, where
   a lagged depth (`"upwind"`) is stable only for ``\Delta t < \Delta x\, W / (3 |q|)``. The matrix keeps
   the sign pattern of an M-matrix but is no longer symmetric.
3. **Solve.** Solve the sparse five-point system for ``h^{k+1}`` with the linear solver
   ([the tutorial](@ref "8. Solvers")).
4. **Limit.** Each row has kinks: the bed, the overburden head, and the boundary head of an outflow-only
   face. An update that crosses one stops at it, so that each solve sees the regime it ends in. Inside the
   regimes the step is limited too: an uplift layer at most doubles per iterate, and the head of an unfilled
   cell sinks below the bed at most twice as deep, or by three quarters of the gap. Under
   `ConsistentGapCoupling` a filled cell's step is capped at ``|\Delta N| \le \max(|N|/2,\, p_o/4)``, since
   on thick ice the gap update is nearly flat in ``N`` and a full Newton step would overshoot (except on the
   filled sheet with englacial storage, which keeps the rows steep). An update
   that reverses the previous one is halved. Optionally the head is relaxed,
   ``h^{k+1} \leftarrow \alpha h^{k+1} + (1 - \alpha) h^k`` (`UnderHeadRelaxation`), or Anderson-accelerated.
5. **Test convergence.** Stop when ``\max |h^{k+1} - h^k| / \max |h^{k+1}| < \mathrm{tol}``
   (or ``\max |h^{k+1} - h^k| < \mathrm{atol}``). Under `"implicit_upwind"` faces, `ConsistentGapCoupling` and
   `UnfilledCavities` (the defaults), with no ``N`` clamp that can bind, the water must balance as well: in
   every cell, ``\mathcal{W}_P(h^{k+1})`` plus the change of englacial water must match what its fluxes and
   sources leave, to ``\max(\mathrm{tol}, 1\%)`` of the sum of the magnitudes of the cell's terms (or to
   ``\mathrm{atol}`` metres). A limited update can stall the head without solving the row, and this test
   catches it. Only there do the rows store exactly the water the conductance carries. Under `"implicit_upwind"`, once the head has converged, a face whose donor was frozen on
   its downstream side is moved to its upstream side and the iteration resumes, at most three times per step.

The first iterate is the head extrapolated in time from the last converged steps
(`head_extrapolation_order`, linear by default), limited in the same way.

## Gap update

`FullyImplicitGapScheme` evaluates both the opening and the closure at ``b^{n+1}``:

```math
b^{n+1} = b^n + \Delta t \left( \frac{\dot m}{\rho_i} + \gamma \left(b_r^* - b^{n+1}\right)_+ - C\, b^{n+1} \right),
\qquad \gamma = \frac{|\mathbf{u}_b|}{l_r}, \quad C = A |N|^{n-1} N ,
```

at ``N = N^{n+1}`` and ``\dot m = \dot m^{n+1}``, with ``b_r^*`` the bump height of
[Gap height](@ref). The equation is piecewise linear in ``b^{n+1}``, so it is solved in closed form:

```math
b^{n+1} =
\begin{cases}
\dfrac{b^n + \Delta t\, (\dot m / \rho_i + \gamma b_r^*)}{1 + \Delta t\, (\gamma + C)} & \text{if this is } < b_r^* , \\[2ex]
\max\!\left(\dfrac{b^n + \Delta t\, \dot m / \rho_i}{1 + \Delta t\, C},\; b_r^*\right) & \text{otherwise.}
\end{cases}
```

Each branch is a step of a linear equation ``\dot b = S - R b``. When its rate ``R`` is negative
(``N < 0``, where creep opens the gap), backward Euler would divide by ``1 - \Delta t |R|``, so the update
is instead the exact solution over the step, ``b^n e^{|R|\Delta t} + S (e^{|R|\Delta t} - 1)/|R|``. The floor at
``b_r^*`` in the second branch matters only then, and keeps the update continuous. With the creep cutoff
``l_c = b^2/b_c`` the branches below ``b_c`` are quadratics, solved the same way. The result is clamped to
``[b_\min, b_\max]``, and the update holds for any ``\Delta t``.

`ExplicitGapScheme` is forward Euler, ``b^{n+1} = b^n + \Delta t\, \dot b(N^{n+1}, b^n)``, accurate and free
of overshoot only for ``\Delta t\, (\gamma + C) < 1``.

On the filled sheet's dry floor the scheme moves the water instead of the gap: ``W^{n+1}`` is the scheme
applied to ``W^n``, clamped to ``[0, b_\max]``, and the gap is ``b^{n+1} = \max(W^{n+1}, b_\min)``, dry
below the water.

Under `WithDiffusion` the wall-melt term ``\nabla\cdot(\mathcal{D}\nabla b)`` couples the cells, so no cell
can compute its gap alone: the gap is solved as a second, symmetric linear system (SUHMO),

```math
b^{n+1} - \Delta t\, \nabla\cdot(\mathcal{D}\nabla b^{n+1}) = b^n + \Delta t \left( \frac{\dot m}{\rho_i} + \gamma \left(b_r^* - b^{n+1}\right) - C\, \frac{l_c}{b^n}\, b^{n+1} \right).
```

Opening by sliding is implicit where ``\beta > 0``, and closure is implicit where ``N > 0`` (through the
lagged ratio ``l_c / b^n``) and explicit, ``C\, l_c``, where ``N \le 0``. The melt is explicit. ``\mathcal{D}`` is
computed per cell and averaged onto the faces between grounded cells, floored at `D_min` (with
`D_faces = true`, the mean wet gap of the two cells times the mean of their heats, as in SUHMO). The result
is clamped to ``[b_\min, b_\max]``.

The head rows must then store a gap that only the whole system can give. Under `ConsistentGapCoupling` the
gap system is solved again at every Picard iterate, at the current ``N``, ``\dot m`` and ``\mathcal{D}``, and
each head row stores that gap, ``b_\mathrm{next} - W^n``, with its slope in the cell's own head (the
neighbours' new gaps held) as the Newton term. This costs one extra linear solve per iterate. Under
`ExplicitGapCoupling` the head rows instead take the explicit rate of the filled row and add
``-\nabla\cdot(\mathcal{D}\nabla b^n)``, the diffusion of the start-of-step gap, to the right-hand side.

## Water and the budget

Under `UnfilledCavities` the water is advanced with the converged fluxes,

```math
W^{n+1}_P = \max\!\left(0,\; W^n_P + \Delta t \left( \frac{\dot m_P}{\rho_w} + i_{e\to b,P} - (\nabla\cdot\mathbf{q})_P \right) - \left( b_e(h^{n+1}_P) - b_e(h^n_P) \right) \right),
```

so the empty part of the gap is ``b^{n+1} - W^{n+1}``, and any water above the gap is the uplift layer.
On the dry floor the gap update sets ``W^{n+1}`` (above). Otherwise the water is the new gap plus the
uplift layer, ``W^{n+1} = b^{n+1} + (h^{n+1} - h_o)_+``, which under `ConsistentGapCoupling` is
``\mathcal{W}_P(h^{n+1})``.

`water_created(sim)` sums, over the grounded cells, the water each step leaves beyond its sources and
fluxes. Under `ConsistentGapCoupling` it is zero to the tolerance of the Picard iteration; under
`ExplicitGapCoupling` it is percents of the input or more.

## Melt on the grid

Each heat term of ``\dot m`` in a cell is built from the cell's four faces. The flow terms average the
face products over the two faces in each direction,

```math
Q_{\mathrm{diss},P} = \rho_w g \left| \sum_f \tfrac12\, q_f (\nabla\hat{h})_f \right|, \qquad
Q_{\mathrm{sens},P} = c_t c_w \rho_w \sum_f \tfrac12\, q_f (\nabla p_w)_f ,
```

except that a face to an `OCEAN` or `LAND` cell is credited in full to the grounded cell, so the heat of
the outflow is not lost. The frictional heat averages the faces' ``u_b \tau_b`` in the same way, with
weight ``\tfrac12`` on every face, ``Q_{b,P} = \left|\sum_f \tfrac12\, u_{b,f}\, \tau_{b,f}\right|``.

With `limit_freeze_on` (the default) the melt rate is floored at the freeze-on that the gap can supply in
the step,

```math
\dot m \ge -\rho_i \max\!\left(0,\; \frac{b^n - b_\min}{\Delta t} + \beta |\mathbf{u}_b| - C\, l_c\right),
```

with ``\beta`` and ``l_c`` evaluated where the gap scheme evaluates them (at ``b_\min`` for the fully
implicit scheme, at ``b^n`` for the explicit one). Under `UnfilledCavities` and on the dry floor the
freeze-on is also limited by the water the cell holds, ``(\rho_w / \rho_i)\, W^n / \Delta t``.
