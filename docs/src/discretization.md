# Discretization

This page shows how the equations of [Model equations](@ref) are solved on the grid. Space is
discretized by finite volumes on a regular grid, and time by backward Euler. Each time step first solves
the head equation by a Picard iteration and then moves the gap. Unless said otherwise, the page describes
the defaults, which are `GlaDS2Sheet()`, `ConsistentGapCoupling`, `"implicit_upwind"` faces and
`FullyImplicitGapScheme`. The safeguards that make the iteration robust, and the bookkeeping around the
solve, are in [Further numerical details](@ref).

## Grid

`Grid(nx, ny, lx, ly)` has ``n_x \times n_y`` cells with centres
``\mathbf{x}_{ij} = ((i-1)\Delta x, (j-1)\Delta y)``, where ``\Delta x = l_x / (n_x - 1)`` and
``\Delta y = l_y / (n_y - 1)``, so ``l_x`` and ``l_y`` are the distances between the outermost centres. The
fields ``b``, ``W``, ``h``, ``p_w``, ``N`` and ``\dot m`` live at the cell centres. The transmissivity ``K``,
the Reynolds number ``\mathrm{Re}`` and the flux ``q`` live on the faces between them (a staggered grid, where
`q_x` has size ``(n_x + 1) \times n_y`` and `q_y` has size ``n_x \times (n_y + 1)``). The face formulas below
are written for the ``x`` face ``i - \tfrac12`` between cells ``i - 1`` and ``i``. The ``y`` faces are the
same with ``\Delta y``.

## [Time step](@id TimeStep)

Integrating the [Water mass balance](@ref) over a cell ``P`` and dividing by its area gives one ordinary
differential equation per cell for its total water thickness,

```math
\frac{\mathrm d}{\mathrm dt}\left( W_P + b_{e,P} \right) + \sum_f \frac{\sigma_f\, q_f}{\Delta_f}
  = \frac{\dot m_P}{\rho_w} + i_{e\to b,P} ,
```

where the sum runs over the four faces of the cell, ``\sigma_f = +1`` on its east and north faces and
``-1`` on its west and south faces, and ``\Delta_f`` is ``\Delta x`` or ``\Delta y``. Here ``W_P`` and ``b_{e,P}``
are averages over the cell. The cell's edges are fixed, so the averages depend on time only, and the time
derivative becomes an ordinary one, ``\mathrm d/\mathrm dt``. With the two-point flux of
[Face flux](@ref FaceFlux), every face term takes the same form, ``\sigma_f q_f / \Delta_f = K_f (\hat{h}_P - \hat{h}_F) / \Delta_f^2``,
with ``F`` the neighbour across face ``f``. Integrating from ``t^{(m)}`` to ``t^{(m+1)} = t^{(m)} + \Delta t``, the
derivative gives the change of the total water thickness over the step exactly. Backward Euler approximates the
other time integrals by ``\Delta t`` times their integrand at the end of the step, so that

```math
\frac{W_P^{(m+1)} - W_P^{(m)}}{\Delta t} + \frac{b_{e,P}^{(m+1)} - b_{e,P}^{(m)}}{\Delta t}
  + \sum_f \frac{K_f}{\Delta_f^2} \left( \hat{h}_P - \hat{h}_F \right)
  = \frac{\dot m_P}{\rho_w} + i_{e\to b,P} ,
```

with ``K_f``, ``\hat{h}``, ``\dot m`` and ``i_{e\to b}`` at ``t^{(m+1)}``. The scheme is first-order accurate in
time. In space the two-point difference is second-order accurate, and the accuracy of ``K_f`` depends on the
face rule. ``W^{(m)}_P`` and ``b^{(m)}_{e,P}`` are the water and the englacial water that the previous step ended
with. A few quantities are held at their start-of-step values instead of at ``t^{(m+1)}``, as listed below.


A step ``m \to m+1`` of length ``\Delta t`` (`step!`) does four things.

1. From the gap ``b^{(m)}`` and the water ``W^{(m)}`` at the start of the step it computes the
   overburden ``p_o``, the jumps of the gap across the faces (for ``\hat{h} = h + b``) and the freeze-on
   limit. These, and the opening coefficient ``\beta`` and creep length ``l_c`` left by the previous step,
   are held for the whole step. A coupled ice model may have changed ``z_s``, ``z_b`` and
   ``\mathbf{u}_b`` before this.
2. The head equation is solved for ``h^{(m+1)}`` by Picard iteration. Every field that
   depends on the head (``p_w``, ``N``, ``\mathbf{q}``, ``\mathrm{Re}``, ``\boldsymbol{\tau}_b`` and
   ``\dot m``) is recomputed at every iterate, so at convergence all of them are evaluated at ``t^{(m+1)}``.
3. The gap scheme ``\mathcal{G}`` moves the gap,
   ``b^{(m+1)} = \mathcal{G}(b^{(m)}; N^{(m+1)}, \dot m^{(m+1)})``.
4. The step sets ``W^{(m+1)}`` and refreshes ``\beta`` and ``l_c`` from ``b^{(m+1)}`` for the
   next step.

The head equation contains ``\partial b / \partial t``, which step 3 has not computed yet, so step 2 must
assume a value for it. The gap coupling decides which. Under `ConsistentGapCoupling` the head rows
assume exactly ``\mathcal{G}``, so steps 2 and 3 store the same water and the water budget closes at any
``\Delta t``. Under `ExplicitGapCoupling` (Shakti's original scheme) they assume
``b^{(m+1)} - b^{(m)} = \Delta t\, \dot b(N^{(m+1)}, b^{(m)})``. This differs from ``\mathcal{G}`` where creep is fast
(``\Delta t A |N|^n > 1``), and the difference is water created or lost
([Gap coupling](@ref GapCoupling)).

## [Face flux](@id FaceFlux)

The flux across a face is driven by the difference of the hydraulic head ``\hat{h} = h + b`` between its
two cells, with the gap held at ``b^{(m)}``:

```math
(\nabla\hat{h})_{i-\frac12} = \frac{(h_i + b^{(m)}_i) - (h_{i-1} + b^{(m)}_{i-1})}{\Delta x} .
```

Under `BedPotential` the ``b^{(m)}`` terms are dropped. On a face to an `OCEAN` or `LAND` cell the boundary
side's gap counts as 0, since its fixed head is already the level of its water.

The laminar transmissivity of the face is

```math
K_{0,f} = \frac{g}{12\nu} \, \overline{W^3}_f ,
```

where ``\overline{W^3}_f`` combines the two cells' water thicknesses as `k_face_choice` says:

| `k_face_choice` | ``\overline{W^3}_f`` |
|:----------------|:---------------------|
| `"implicit_upwind"` (default on the free-surface sheets) | the donor's ``W^3``, with ``W`` the donor's water at the end of the step |
| `"upwind"` | the donor's ``W^3``, with ``W`` the donor's water at the start of the step |
| `"arithmetic"` (default on the filled sheet) | ``(W_{i-1}^3 + W_i^3)/2`` |
| `"harmonic"` | ``2 W_{i-1}^3 W_i^3 / (W_{i-1}^3 + W_i^3)`` |
| `"gap_mean"` | ``\left((W_{i-1} + W_i)/2\right)^3``, the face rule of SUHMO |

The donor is the cell with the higher ``\hat{h}``, the one the water flows out of. Taking the donor's water thickness
keeps an empty cell from draining water it does not have, which is why the averaged faces are refused on
the free-surface sheets. Within a step, a face whose donor has reversed four times keeps its donor, so
the iteration cannot cycle on it. A face to an `OCEAN` or `LAND` cell takes the grounded cell's ``W``, and
a face to `OTHER_BASIN`, `FROZEN_BED` or the edge of the grid has ``K_f = 0``.

With the [free-surface layer](@ref "Free-surface layer") on (the default), each cell's ``W^3`` is first multiplied by
``1 + (f_K - 1)\, w``, its weight ``w`` taken at the head iterate, and the face's ``\omega`` becomes
``\omega_f = \omega\,(1 + (f_{\omega} - 1)\, w_f)``. ``w_f`` is the weight of the cell whose transmissivity the face takes
(the donor, or the grounded cell at a boundary face), or the mean of the two cells' weights for a mean of their
transmissivities. Otherwise ``\omega_f = \omega``.

The turbulent correction follows from the flux law. With ``D_f = K_{0,f} |(\nabla\hat{h})_f| / \nu``, the
Reynolds number ``\mathrm{Re} = |q|/\nu`` solves ``\omega_f \mathrm{Re}^2 + \mathrm{Re} - D_f = 0``, so

```math
\mathrm{Re}_f = \frac{2 D_f}{1 + \sqrt{1 + 4 \omega_f D_f}}, \qquad
K_f = \frac{K_{0,f}}{1 + \omega_f \mathrm{Re}_f}, \qquad
q_f = -K_f (\nabla\hat{h})_f .
```

This form of the root avoids cancellation as ``\omega \to 0`` and gives ``|q_f| = \nu \mathrm{Re}_f``
exactly, with no lag between ``q`` and ``\mathrm{Re}``. The same ``K_f`` is assembled into the matrix and
used by every diagnostic (``\mathbf{q}``, ``\mathrm{Re}``, ``\dot m``, ``\mathcal{D}``), so the water that
the solve conserves and the water that melts the ice are the same.

## [Boundary cells and faces](@id BoundaryCellsAndFaces)

A `LAND` or `OCEAN` cell has a fixed reduced head ``h_D`` at its centre, with the value of
[Boundary conditions](@ref). Since the head sits at the cell centre and not on the face, a face to land or
ocean spans the same distance ``\Delta_f`` between two cell centres as an interior face. The boundary cell
holds no gap, ``b = 0``, so its hydraulic head is ``h_D`` itself, and the gap jump across the face is the
grounded cell's gap, ``\pm b^{(m)}_P``.

The boundary faces are outflow-only by default (`outflow_only_land`, `outflow_only_ocean`), and `ShaktiSheet(original = true)`
keeps them open in both directions. The flux out of
a grounded cell ``P`` through such a face is

```math
q_f = \frac{K_f}{\Delta_f} \max\left(0,\; h_P + b^{(m)}_P - h_D\right) ,
```

so the face is closed, ``K_f = 0``, while the boundary head lies above the grounded cell's water level
``h + b``, and the boundary drains the ice but never feeds it. As a function of ``h_P`` the flux is zero below
``h_D - b^{(m)}_P`` and linear above, which puts a kink in the row of ``P``. Whether each such face is open is
therefore decided again from the head at every Picard iterate, and an update that crosses the kink stops at
it ([Picard iteration](@ref PicardIteration), step 4).

With the option off, the face is an ordinary open face. Its flux ``K_f (h_P + b^{(m)}_P - h_D) / \Delta_f`` may
then be negative, and water enters the bed from the boundary, for example seawater beneath grounded ice.

## [The discrete head equation](@id HeadEquation)

For a cell ``P`` with neighbours ``F`` across its four faces ``f``, the divergence of the face fluxes is

```math
(\nabla\cdot\mathbf{q})_P = \frac{q_{i+\frac12} - q_{i-\frac12}}{\Delta x} + \frac{q_{j+\frac12} - q_{j-\frac12}}{\Delta y} .
```

Integrating the mass balance over the cell and over the step gives the row of cell ``P``:

```math
\sum_f \frac{K_f}{\Delta_f^2} \left( h_P - h_F \right)
  + \frac{b_e(h_P) - b^{(m)}_{e,P}}{\Delta t}
  = \frac{\dot m_P}{\rho_w} + i_{e\to b,P}
  - \frac{\mathcal{W}_P(h_P) - W^{(m)}_P}{\Delta t}
  - \sum_f \frac{K_f}{\Delta_f^2} \left( b^{(m)}_P - b^{(m)}_F \right) ,
```

where ``\Delta_f`` is ``\Delta x`` or ``\Delta y``. On the left are the net outflow and the englacial storage.
On the right are the sources, the change of the water at the bed, and the part of the outflow driven by the
known gap jumps. ``\mathcal{W}_P(h)``, the water-thickness function, is the water at the bed, the water thickness
``W_P``, at the end of the step if the head of the cell is ``h``. The total water thickness of the cell is
``\mathcal{W}_P(h) + b_e(h)``, the water at the bed together with the englacial water. By the closure of [Three regimes of the sheet](@ref),

```math
\mathcal{W}_P(h) =
\begin{cases}
\mathcal{G}_P(p_o) - (z_b - h) & h < z_b \quad \text{(unfilled)} \\
\mathcal{G}_P\!\left(p_o - \rho_w g (h - z_b)\right) & z_b \le h \le h_o \quad \text{(filled)} \\
\mathcal{G}_P(0) + (h - h_o) & h > h_o \quad \text{(uplifted)}
\end{cases}
```

with ``\mathcal{G}_P(N)`` the gap after the step at effective pressure ``N``. The three regimes agree at
``h = z_b`` and ``h = h_o``, so a cell fills, pressurizes and lifts the ice continuously. In the unfilled
and uplifted regimes ``\mathcal{W}_P`` grows with ``h`` with slope 1, which puts ``1/\Delta t`` on the
diagonal. In the filled regime the slope is ``\rho_w g\, |\mathrm d\mathcal{G}_P / \mathrm dN|`` (closure
grows with ``N``), computed by a central difference. On the dry floor the filled regime
moves the water ``W^{(m)}`` instead of the gap, with no lower clamp at ``b_{\min}``, so ``\mathcal{W}_P`` keeps a
non-zero slope in a drained cell.

### [Englacial storage](@id EnglacialStorage)

The englacial water ``b_e = e_v\, p_w / (\rho_w g)`` is the water held in the ice above the bed, per unit area of
bed. By default ``e_v = 0`` and there is none. With unfilled cavities and uplift ``p_w`` lies between 0 and
``p_o``, so

```math
b_e(h) = e_v \min\!\left( \max(h - z_b,\, 0),\; \frac{p_o}{\rho_w g} \right), \qquad
\frac{\mathrm d b_e}{\mathrm d h} =
\begin{cases}
e_v & \text{filled} \\
0 & \text{unfilled or uplifted} .
\end{cases}
```

Below the bed and above the overburden head ``b_e`` stays constant, and any further change of the head empties
the gap or lifts the ice instead. Without unfilled cavities the lower limit is dropped, and without uplift the
upper one.

The row stores the change over the step, ``\big(b_e(h^{(m+1)}) - b^{(m)}_e\big)/\Delta t``, the
backward-Euler step of ``\partial b_e / \partial t``. Linearized about the iterate, it adds
``(\mathrm d b_e/\mathrm d h)/\Delta t`` to the diagonal.

The start-of-step value ``b^{(m)}_e`` is the value the last step ended with, kept in the `State` field `b_e`.
It is not recomputed from ``h^{(m)}``, because ``p_o`` changes between steps with the gap and the surface.
Recomputing it would change the englacial water of an uplifted cell by ``e_v\, \Delta p_o / (\rho_w g)`` without any
flux. With the stored value, the drop in englacial water appears in the englacial term of the row,
``\big(b_e(h^{(m+1)}) - b^{(m)}_e\big)/\Delta t``. The row then moves that water into the sheet within the step,
where the budget counts it, so no water is lost.

With ``e_v > 0`` the total water thickness gradient of a filled cell,
``\mathrm d(\mathcal{W} + b_e)/\mathrm dh``, is at least ``e_v``, so the diagonal of every row stays
positive even when all its faces are closed. Closed cells then take the ordinary row ([Closed cells](@ref ClosedCells)).

On the filled sheet the englacial storage also switches off the filled cap of
[Limiting the head iterate](@ref LimitingTheHead). Take one filled cell with a face of transmissivity ``K_f`` to a
neighbour at distance ``\Delta x``. Without englacial storage, linearizing its water-thickness function ``\mathcal{W}`` about the iterate and solving the row gives the step

```math
h^{(k+1)} - h^{(k)} = -\frac{\mathcal{R}(h^{(k)})}{(\mathrm d\mathcal{W}/\mathrm dh)/\Delta t + K_f/\Delta x^2},
```

the imbalance ``\mathcal{R}`` of the row divided by how strongly the row responds to the head. In a filled cell
``\mathcal{W} = \mathcal{G}(N)``. On thick ice near ``N = p_o`` creep closes the gap almost completely within the
step, whatever the exact ``N``, and a nearly closed gap can barely change. Under 1000 m of ice, with a time step of one day and no
sliding, a gap that the update closes to 1 mm gives ``\mathrm d\mathcal{W}/\mathrm dh \approx 3 \times 10^{-6}``, so where ``K_f`` is small too, an imbalance of 10 mm of water over the step
gives a step of about 3000 m of head. That is more than three times the 910 m between the bed and the overburden
head, while the head that balances the water lies about 500 m away. As the head rises, ``N`` falls, creep slows and
``\mathrm d\mathcal{W}/\mathrm dh`` grows quickly, which the slope at the iterate does not see. The cap stops this overshoot. With
``e_v > 0`` the gradient in the step is the total water thickness gradient ``\mathrm d\mathcal{W}/\mathrm dh + e_v``, about 30 times larger at ``e_v = 10^{-4}``. This straight part
dominates where ``\mathcal{W}`` is flat, so the step comes out of the right size, and the cap would only cut good
steps short and slow the iteration down.

### [Gap couplings](@id GapCouplings)

The head rows are solved before the gap update has run ([Time step](@ref TimeStep)), so each row must assume what
the gap will be at the end of the step. The gap coupling decides which gap that is.

Under `ConsistentGapCoupling` (the default) the row assumes the gap that the gap update ``\mathcal{G}_P`` itself
will give. The row then stores the water-thickness function ``\mathcal{W}_P(h)`` built from ``\mathcal{G}_P``
([The discrete head equation](@ref HeadEquation)). In the three regimes

| regime | water-thickness function ``\mathcal{W}_P(h)`` | gradient ``\mathrm d\mathcal{W}_P/\mathrm dh`` |
|:-------|:----------------------------------------------|:-----------------------------------------------|
| unfilled, ``h < z_b`` | ``\mathcal{G}_P(p_o) - (z_b - h)`` | 1 |
| filled | ``\mathcal{G}_P(N)``, ``N = p_o - \rho_w g (h - z_b)`` | ``\rho_w g \max\!\left(0,\, -\mathrm d\mathcal{G}_P / \mathrm dN\right)`` |
| uplifted, ``h > h_o`` | ``\mathcal{G}_P(0) + (h - h_o)`` | 1 |

The unfilled and uplifted rows are linear in ``h``. They evaluate ``\mathcal{G}_P`` at ``N = p_o`` and ``N = 0``,
which are the values of the filled row at ``h = z_b`` and at ``h = h_o``, so the rows join continuously. The filled
row is nonlinear in ``h`` and is linearized about the iterate, ``\mathcal{W}_P(h) \approx \mathcal{G}_P(N^{(k)}) +
(\mathrm d\mathcal{W}_P/\mathrm dh)(h - h^{(k)})``, with the gradient taken at ``N^{(k)}``. That gradient is derived in
[Derivative of the gap update](@ref GapDerivative). The melt rate in ``\mathcal{G}_P`` is recomputed from the
fluxes at every iterate, so the row conserves the water of the melt at the converged head. Only the transmissivity
holds the melt of the start of the step ([Melt in the transmissivity](@ref MeltInTransmissivity)). The transmissivity
sets the size of a face flux, and both cells of the face take the same flux, so this lag changes the accuracy of
the head but creates or loses no water.

Under `ExplicitGapCoupling` the row instead assumes that the gap changes at the rate it has for the start-of-step
gap and the new effective pressure,

```math
b^{(m+1)} - b^{(m)} = \Delta t \left( \frac{\dot m}{\rho_i} + \beta(b^{(m)})\, |\mathbf{u}_b| - C(N)\, l_c(b^{(m)}) \right),
\qquad C(N) = A |N|^{n-1} N .
```

The change of the water thickness over the step, ``(\mathcal{W}_P - W^{(m)}_P)/\Delta t``, then becomes

```math
\frac{b^{(m)}_P - W^{(m)}_P}{\Delta t} + \frac{\dot m_P}{\rho_i} + \beta(b^{(m)}_P)\, |\mathbf{u}_b|_P - C(N_P)\, l_c(b^{(m)}_P) ,
```

with ``N_P = p_o - \rho_w g (h_P - z_b)`` in a filled cell. An unfilled cell has ``N_P = p_o`` and adds
``(h_P - z_b)/\Delta t``, the change of its empty part. An uplifted cell has ``N_P = 0`` and adds
``(h_P - h_o)/\Delta t``, the change of its uplift layer. The first term, ``(b^{(m)}_P - W^{(m)}_P)/\Delta t``, is the
empty part of the gap (positive) or the uplift layer (negative) that the last step left, which the row fills or
drains first. It is zero in a cell that ended the last step filled. The filled row ignores the clamps of the gap
update at ``b_{\min}`` and ``b_{\max}``. Where a clamp binds, the water the clamp keeps out of the gap is lost at
``b_{\max}`` and created at ``b_{\min}``. The unfilled and uplifted rows set the rate to zero where the gap is held
at a clamp.

The creep term changes steeply with ``N``, so it is linearized about the iterate (Newton). Since
``\mathrm dC/\mathrm dh = -\rho_w g\, n A |N|^{n-1}``,

```math
C(N)\, l_c \approx C(N^{(k)})\, l_c - g_d \left( h - h^{(k)} \right), \qquad g_d = n \rho_w g A |N^{(k)}|^{n-1} l_c ,
```

which puts ``g_d`` on the diagonal and ``C(N^{(k)})\, l_c + g_d h^{(k)}`` on the right-hand side. For
``\Delta t R \ll 1``, ``g_d`` is the limit of the gradient ``\mathrm d\mathcal{W}_P/\mathrm dh`` of the filled row of
the consistent coupling ([Derivative of the gap update](@ref GapDerivative)), so the two couplings agree at short
time steps.

Paired with `FullyImplicitGapScheme`, the explicit coupling no longer agrees with the gap update, and the step
creates water. Take a cell whose gap stays in one case of the update over the step, with no clamp binding, and
write that case as ``\mathrm db/\mathrm dt = S - R\, b`` ([Accuracy of the gap update](@ref GapAccuracy)). The row
assumes that the gap changes by ``\Delta t\, (S - R\, b^{(m)})``, while the update changes it by
``\Delta t\, (S - R\, b^{(m)})/(1 + \Delta t R)``. The difference is water that the step creates
([Water and the budget](@ref WaterAndBudget)). It is positive where the gap closes, since the row sends away the
full explicit closure but the update closes the gap less. For ``\Delta t R \ll 1`` it is of order ``\Delta t^2``
per step, the error of an operator split. For ``\Delta t R \gg 1`` the change of the updated gap stays bounded,
tending to ``S/R - b^{(m)}``, while the change the row assumes grows with ``\Delta t``. Nearly all of the closure
the row assumed is then created water. Once ``\Delta t R > 1 + \Delta t\, S / b^{(m)}`` the row even assumes that
more water leaves the gap than it holds. Under `ConsistentGapCoupling` the rows and the update use the same
``\mathcal{G}_P``, so no water is created this way at any ``\Delta t``.

The explicit coupling is the scheme of the original Shakti model [Sommers2018](@cite). In a filled cell without
englacial storage, with ``W = b`` and with ``h`` in place of ``\hat{h}`` in the fluxes, the row is their head
equation,

```math
\sum_f \frac{K_f}{\Delta_f^2} \left( h_P - h_F \right)
  = \dot m_P \left( \frac{1}{\rho_w} - \frac{1}{\rho_i} \right) - \beta_P |\mathbf{u}_b|_P
  + A_P |N_P|^{n-1} N_P\, b^{(m)}_P + i_{e\to b,P} .
```

The original scheme then advances the gap with `ExplicitGapScheme` at the converged ``N``, so its head equation
and its gap update agree, but the gap update is stable only for ``\Delta t\, R < 2``
([Accuracy of the gap update](@ref GapAccuracy)). `ShaktiSheet(original = true)` selects `FilledCavities`,
`NoUplift`, `BedPotential`, `ExplicitGapCoupling` and boundary faces open in both directions. With `ExplicitGapScheme` and `"arithmetic"` faces added,
Shakti.jl solves these equations with their time discretization, on a finite-volume grid in place of their finite
elements. The original code solves for the head at the vertices of a triangular mesh and holds the gap, the
Reynolds number and hence ``K`` constant on each triangle. Two details still differ. Their transmissivity is
constant on each triangle, built from the gap of that triangle, while ours is the mean of the two cells' laminar
transmissivities on each face. Their Reynolds number is updated from the flux of the previous iterate, while ours
solves the quadratic of [Face flux](@ref FaceFlux) on every face. Both give the same Reynolds number once the
iteration has converged.

### [Special rows](@id SpecialRows)

Some rows are special:

- A grounded cell whose four faces are all closed, with ``e_v = 0``, exchanges no water. Its water at the
  end of the step, ``W^{(m)} + \Delta t (\dot m / \rho_w + i_{e\to b})``, is known, and ``\mathcal{W}_P(h)`` equal
  to it is solved for ``h`` exactly, by bisection, or under `ExplicitGapCoupling` by inverting the creep
  law in closed form.
- An `OCEAN` or `LAND` neighbour's head is known, so its term moves to the right-hand side. This keeps the
  matrix symmetric wherever the face scheme allows it. The face's transmissivity stays on the cell's
  diagonal, which makes the matrix positive definite next to an open boundary face.
- `OCEAN` and `LAND` cells have the row ``h_P = `` their boundary head, and `OTHER_BASIN` and
  `FROZEN_BED` cells the row ``h_P = `` their current head.

### [Clamps on the effective pressure](@id ClampsOnN)

``N`` can be clamped to ``[N_{\min}, N_{\max}]``, for the whole grid (`N_min` and `N_max` of `ModelParameters`) or
for single cells (`CellNClamping`, with its own bounds in each cell it lists). By default there is no clamp,
``N_{\min} = -\infty`` and ``N_{\max} = \infty``. The clamps are kept as a safeguard for the filled sheet.

From the head, every grounded or frozen-bed cell gets the water pressure ``p_w = \rho_w g (h - z_b)``, floored at 0 with unfilled
cavities and capped at ``p_o`` with uplift. Its effective pressure is then

```math
N = \operatorname{clamp}\!\left(p_o - p_w,\; N_{\min},\; N_{\max}\right),
```

and `CellNClamping` clamps it again in its own cells. The head rows and the gap update both use this clamped
``N``, while ``p_w`` is not clamped. Where a clamp binds, ``N \ne p_o - p_w`` and ``N`` stays at the bound whatever
the head of the cell. The gap the row assumes, and with it the water of the cell at the end of the step, then no
longer changes with the head. The Newton term of the row, ``(\mathrm d\mathcal{W}_P/\mathrm dh)/\Delta t`` or
``g_d`` ([Gap couplings](@ref GapCouplings)), still appears on the diagonal and, times ``h^{(k)}``, on the
right-hand side. The two cancel once the iteration has converged, so the term only damps the iteration, and the
head of the cell is set by its face fluxes.

With unfilled cavities or uplift, a clamp tighter than ``[0, \infty)`` is refused when the simulation is set up.
The unfilled row assumes ``N = p_o`` and the uplifted row ``N = 0``, while the gap update would close the gap at
the clamped ``N``. The difference would stay in ``W``, and the ``N`` passed to an ice-flow model would no longer
be ``p_o - p_w``.

Where a clamp binds, the water of the row no longer follows its head, so the water test of
[Picard iteration](@ref PicardIteration) is used only where no clamp can bind. With unfilled cavities and uplift,
``N`` already lies in ``[0, p_o]``, so the clamps that remain allowed never bind and the test stays on. With
unfilled cavities but no uplift, ``p_w`` can exceed ``p_o`` and ``N`` can fall below 0, so even a floor at 0 can
bind. Any finite floor then switches the water test off, whether it binds or not.

## [Picard iteration](@id PicardIteration)

The row of [The discrete head equation](@ref HeadEquation) is nonlinear in ``h``. The head enters through ``N``
in ``\mathcal{G}``, through ``W`` and ``\mathrm{Re}`` in ``K_f``, through ``\mathbf{q}`` and
``\boldsymbol{\tau}_b(N)`` in ``\dot m``, and through the regime of each cell. The row is linearized about the
current iterate ``h^{(k)}`` in one of two ways. A coefficient that depends on ``h`` can be evaluated at
``h^{(k)}`` and held while the linear system is solved, which is a Picard, or fixed-point, iteration. A term
``f(h)`` can instead be replaced by its tangent at ``h^{(k)}``,
``f(h^{(k)}) + (\mathrm df/\mathrm dh)(h^{(k)})\,(h - h^{(k)})``, which is Newton's method. The part
``(\mathrm df/\mathrm dh)(h^{(k)})\, h`` that the tangent adds to the matrix is called a Newton term. Shakti.jl
uses Newton terms for the water-thickness function ``\mathcal{W}_P`` of each row and for the water thickness of
the donor of each face, where the Picard iteration diverges at practical time steps
([Why the water thickness is differentiated](@ref WhyDifferentiate)). It holds ``\mathrm{Re}_f``, ``\dot m`` and
the regimes at ``h^{(k)}``. The solver is called `PicardSolver`, but the iteration is a partial Newton method.

An iterate ``k \to k+1`` does five things.

1. Evaluate ``K_f``, ``\dot m`` and the regime of every cell at ``h^{(k)}``. ``\mathcal{W}_P`` has one formula
   in each regime, unfilled, filled and uplifted ([Gap couplings](@ref GapCouplings)). Each row takes the formula
   of the regime that ``h^{(k)}_P`` lies in and keeps it for the solve, even if ``h^{(k+1)}_P`` ends in another
   regime (step 4 keeps such crossings small). The regime is recomputed from the head at every iterate, so a cell
   that step 4 stopped just inside a new regime takes the formula of that regime in the next solve. The unfilled
   and uplifted formulas are already linear in ``h``. The filled one is linearized about ``h^{(k)}_P``,
   ``\mathcal{W}_P(h) \approx \mathcal{W}_P(h^{(k)}) + (\mathrm d\mathcal{W}_P/\mathrm dh)(h^{(k)})\,(h - h^{(k)})``.
2. Under `"implicit_upwind"`, add the Newton terms of the donors' water thickness
   ([Newton terms of the donors](@ref DonorNewtonTerms)).
3. Solve the sparse five-point system for ``h^{(k+1)}`` with the linear solver
   ([the tutorial](@ref "8. Solvers")).
4. Damp and limit the change of the head. The optional under-relaxation and the stall damping scale the change
   first. The limits then act cell by cell, so that an update that crosses a kink of its row stops at the kink and
   the steps inside a regime stay bounded ([Limiting the head iterate](@ref LimitingTheHead)).
5. Recompute ``p_w``, ``N``, the face fluxes and ``\dot m`` at the new head, and stop when the tests of
   [Convergence tests](@ref ConvergenceTests) pass.

The terms held at ``h^{(k)}`` do not change the head the iteration converges to. With ``\boldsymbol{\mathcal{R}}``
the residuals of the rows (left-hand side minus right-hand side) and ``\mathbf{A}`` the matrix of an iterate, each
linear system is built so that

```math
\mathbf{A}\, \left(\mathbf{h}^{(k+1)} - \mathbf{h}^{(k)}\right) = -\boldsymbol{\mathcal{R}}(\mathbf{h}^{(k)}) .
```

Once the head stops changing, the residuals are zero and the nonlinear rows are solved exactly. The held terms,
the derivatives of ``\mathrm{Re}_f``, ``\dot m`` and the regimes, only make ``\mathbf{A}`` differ from the Jacobian
of ``\boldsymbol{\mathcal{R}}``, its matrix of derivatives in ``\mathbf{h}``. They affect how fast the iteration
converges, not where it ends.

### [Newton terms of the donors](@id DonorNewtonTerms)

Under `"implicit_upwind"`, ``K_f`` takes the water thickness of the donor at the end of the step,
``W_d = \mathcal{W}_d(h^{(k)})``, with its melt opening from the start of the step
([Melt in the transmissivity](@ref MeltInTransmissivity)). The outflow of the face then depends on the head of
the donor through ``W_d``. Holding ``W_d`` at the iterate makes the iteration diverge at practical time steps
([Why the water thickness is differentiated](@ref WhyDifferentiate)), so the outflow is linearized in the donor's
head ``h_d`` at a fixed head gradient,

```math
|q_f|(h_d) \approx |q_f|(h^{(k)}_d) + J_f \left( h_d - h^{(k)}_d \right), \qquad J_f = \frac{\mathrm d|q_f|}{\mathrm dh_d} .
```

The row of the donor gains ``J_f/\Delta_f`` on its diagonal, and the row of the receiver gains ``-J_f/\Delta_f``
in the donor's column. The right-hand sides gain the same terms times ``h^{(k)}_d``, so both cancel at
``h = h^{(k)}``.

The gradient of the donor's water thickness in its head, ``\mathrm d\mathcal{W}_d/\mathrm dh_d``, is the one of
its own row ([Gap couplings](@ref GapCouplings)),

| regime of the donor | ``\mathrm d\mathcal{W}_d/\mathrm dh_d`` |
|:--------------------|:----------------------------------------|
| unfilled | 1 |
| uplifted | 1 |
| filled, `ConsistentGapCoupling` | ``\rho_w g \max\!\left(0,\, -\mathrm d\mathcal{G}_d/\mathrm dN\right)``, by a central difference in ``N`` |
| filled, `ExplicitGapCoupling` | 0, since ``\mathcal{W}_d`` is the start-of-step gap |

Where ``\mathcal{W}_d`` would fall below 0, the water thickness and its gradient are both 0.

Without the free-surface layer the head of the donor changes only ``W_d``. The flux is ``|q_f| = \nu \mathrm{Re}_f``,
with ``\mathrm{Re}_f`` the root of ``\omega \mathrm{Re}^2 + \mathrm{Re} - D_f = 0``, ``D_f = K_{0,f}
|(\nabla\hat{h})_f| / \nu`` and ``K_{0,f} = g W_d^3/(12\nu)`` ([Face flux](@ref FaceFlux)). Differentiating the
quadratic in ``h_d`` at a fixed head gradient gives

```math
\left( 1 + 2\omega\mathrm{Re}_f \right) \frac{\mathrm d\mathrm{Re}_f}{\mathrm dh_d} = \frac{\mathrm dD_f}{\mathrm dh_d}
  = \frac{3 D_f}{W_d}\, \frac{\mathrm d\mathcal{W}_d}{\mathrm dh_d} .
```

With ``\nu D_f = K_{0,f} |(\nabla\hat{h})_f| = (1 + \omega\mathrm{Re}_f)\, |q_f|``, multiplying by ``\nu`` gives

```math
J_f = \frac{3 |q_f| \, (1 + \omega\mathrm{Re}_f)}{W_d\, (1 + 2\omega\mathrm{Re}_f)}\, \frac{\mathrm d\mathcal{W}_d}{\mathrm dh_d} \ge 0 ,
```

which is ``3|q_f|/W_d`` times ``\mathrm d\mathcal{W}_d/\mathrm dh_d`` in laminar flow and half of that in fully
turbulent flow.

With the free-surface layer the head of the donor also changes its weight ``w_d``. The weight enters the flux law
twice ([Face flux](@ref FaceFlux)), in the laminar transmissivity and in the turbulent coefficient,

```math
K_{0,f} = \frac{g}{12\nu} \left( 1 + a\, w_d \right) W_d^3, \qquad \omega_f = \omega \left( 1 + o\, w_d \right),
\qquad a = f_K - 1, \quad o = f_{\omega} - 1 .
```

Inside the ramp of [The free-surface switch](@ref FreeSurfaceSwitch) the weight is ``w = t^2 (3 - 2t)`` with
``t = (z_b - h)/(\beta_{\mathrm{fs}}\, b)``. As the head rises the cell becomes less empty, ``t`` falls and the
weight falls with it,

```math
\frac{\mathrm dw_d}{\mathrm dh_d} = -\frac{6\, t\, (1 - t)}{\beta_{\mathrm{fs}}\, b} \le 0 ,
```

and ``\mathrm dw_d/\mathrm dh_d = 0`` outside the ramp. Differentiating ``\omega_f \mathrm{Re}^2 + \mathrm{Re} - D_f = 0``
with both ``D_f`` and ``\omega_f`` changing gives

```math
\left( 1 + 2\omega_f\mathrm{Re}_f \right) \frac{\mathrm d\mathrm{Re}_f}{\mathrm dh_d}
  = \frac{\mathrm dD_f}{\mathrm dh_d} - \mathrm{Re}_f^2 \frac{\mathrm d\omega_f}{\mathrm dh_d},
\qquad
\frac{\mathrm dD_f}{\mathrm dh_d} = D_f \left( \frac{3}{W_d} \frac{\mathrm d\mathcal{W}_d}{\mathrm dh_d}
  + \frac{a}{1 + a w_d} \frac{\mathrm dw_d}{\mathrm dh_d} \right),
\qquad
\frac{\mathrm d\omega_f}{\mathrm dh_d} = \omega\, o\, \frac{\mathrm dw_d}{\mathrm dh_d} .
```

Multiplying by ``\nu``, with ``\nu D_f = K_f (1 + \omega_f\mathrm{Re}_f) |(\nabla\hat{h})_f|``,

```math
J_f = \frac{1}{1 + 2\omega_f\mathrm{Re}_f} \left[ K_f |(\nabla\hat{h})_f| (1 + \omega_f\mathrm{Re}_f)
  \left( \frac{3}{W_d} \frac{\mathrm d\mathcal{W}_d}{\mathrm dh_d} + \frac{a}{1 + a w_d} \frac{\mathrm dw_d}{\mathrm dh_d} \right)
  - \nu\, \omega\, o\, \mathrm{Re}_f^2\, \frac{\mathrm dw_d}{\mathrm dh_d} \right] .
```

As the head rises, the first term in the round brackets is positive, since thicker water carries more. The second
is negative, since the cell turns from a free-surface layer into a duct, which carries up to ``f_K`` times less
at the same water thickness. The last term is positive, since ``\omega_f`` falls towards ``\omega`` and turbulence resists the
flow less. Where the second term outweighs the others, the outflow falls as the head rises and ``J_f`` is
negative. A negative ``J_f`` shrinks the donor's diagonal and raises the entry of the donor in the receiver's
row, which is ``-K_f/\Delta_f^2 - J_f/\Delta_f``, and once ``J_f < -K_f/\Delta_f`` that entry is positive. The
receiver's head would then fall as the donor's rises, and the matrix would lose the M-matrix pattern below, on
which the absence of spurious oscillations and the linear solvers rely. A negative ``J_f`` is therefore set to 0.
The face then keeps its transmissivity at ``W_d(h^{(k)})`` for the solve, a Picard step for this one dependence,
which can slow the iteration but does not change the head it converges to. A ramp at least as wide as the bound of
[The free-surface switch](@ref FreeSurfaceSwitch), which the default `fs_blend = 0.65` is, keeps ``J_f \ge 0``.

``J_f`` is also 0 where the face carries no flux, where the donor is not grounded, where
``\mathrm d\mathcal{W}_d/\mathrm dh_d = 0``, and where the donor is held on the downstream side of the face
([Donors near a flow divide](@ref FlowDivide)). With ``J_f \ge 0`` the matrix keeps the sign pattern of an
M-matrix, a positive diagonal and no positive entries off it, but it is no longer symmetric
([Linear solvers](@ref LinearSolvers)).

### [Convergence tests](@id ConvergenceTests)

The iteration stops when two tests pass, one on the head and one on the water. The head test asks that the head
has stopped changing,

```math
\frac{\max_P |h^{(k+1)}_P - h^{(k)}_P|}{\max_P |h^{(k+1)}_P|} < \epsilon_h ,
```

with ``\epsilon_h`` = `tol` (``10^{-6}`` by default). The change is the one applied after the limits, divided by
the factors of the stall damping and of the optional under-relaxation, so that a damped iteration does not stop
early. With Anderson acceleration the test uses the accelerator's own residual instead.

The head test alone is not enough, for two reasons. Relative to the largest head, ``10^{-6}`` is 4.5 mm on a bed
at 4500 m, too loose where a millimetre of head is a millimetre of water. And where a limit holds an iterate
almost still, the head test passes although the row is not solved. Under `"implicit_upwind"` faces,
`ConsistentGapCoupling` and `UnfilledCavities` (the defaults), with no ``N`` clamp that can bind
([Clamps on the effective pressure](@ref ClampsOnN)), the water must therefore balance as well. In every grounded
cell the residual

```math
r_P = \mathcal{W}_P(h^{(k+1)}) + b_{e,P}^{(k+1)} - W_P^{(m)} - b_{e,P}^{(m)}
    - \Delta t \left( \frac{\dot m_P}{\rho_w} + i_{e\to b,P} - (\nabla\cdot\mathbf{q})_P \right)
```

is the change of the total water thickness over the step minus the water that the sources and the face fluxes
bring. It must satisfy ``|r_P| < \epsilon_w\, \Sigma_P`` with ``\epsilon_w = \max(\epsilon_h, 10^{-3})`` and

```math
\Sigma_P = \mathcal{W}_P(h^{(k+1)}) + \left| b_{e,P}^{(k+1)} - b_{e,P}^{(m)} \right| + W_P^{(m)}
    + \Delta t \left( \left| \frac{\dot m_P}{\rho_w} \right| + \left| i_{e\to b,P} \right| \right) ,
```

the size of the cell's own water and sources. The face fluxes are left out of ``\Sigma_P``, since a cell the
water flows through can carry many times its own input. A head error far below the head tolerance can then
change its water thickness by a large fraction of that input. The residual ``r_P`` is also the difference
between two water thicknesses at the end of the step, the one the head gives through ``\mathcal{W}_P`` and the
one the budget of [Water and the budget](@ref WaterAndBudget) gives. Where the test passes, the head and the
budget agree on the water of the cell.

Two exceptions keep the test from failing on round-off and on donor switching. A cell also passes when
``|r_P|`` is within 100 machine epsilons of the sum of all its terms, fluxes included. And after ten iterates
whose head has converged but whose water has not, the test is relaxed to ``10^{-2}`` of all the cell's terms,
fluxes included. This admits cells next to a face whose donor flips, whose residual levels off near ``10^{-3}``
of the throughflow.

Both tests can instead be made absolute with `atol > 0` (off by default), which replaces the relative ones. The
head test then asks for ``\max_P |h^{(k+1)}_P - h^{(k)}_P|`` below `atol` in metres of head, and the water test
for ``\max_P |r_P|`` below `atol` in metres of water.

Under `"implicit_upwind"`, once the head test passes, a face whose donor was held on its downstream side is moved
to its upstream side and the iteration resumes, at most three times per step
([Donors near a flow divide](@ref FlowDivide)). A step that does not converge within the maximum number of
iterates is accepted and counted, or retried at ``\Delta t/2`` under the adaptive time step
([Time-step control](@ref TimeStepControl)). The first iterate of a step is the head extrapolated in time from
the last converged steps ([The first iterate](@ref FirstIterate)).

### [Kinematic waves](@id KinematicWave)

A kinematic wave appears wherever the flux of a quantity ``W`` depends only on how much of it is there, and not
on how it changes in space, ``\partial W/\partial x``. With a flux ``F(W)`` the balance ``\partial W/\partial t + \partial F(W)/\partial x = 0`` becomes, by the
chain rule,

```math
\frac{\partial W}{\partial t} + c\, \frac{\partial W}{\partial x} = 0, \qquad c = \frac{\mathrm dF}{\mathrm dW} ,
```

which carries ``W`` along at the speed ``c``. A bump of water then moves downstream at that speed, and the bump is
the wave. It is called kinematic because its speed follows from the relation between the flux and the amount
alone, with no force or momentum balance.

In the sheet the wave comes from water on a sloping bed. An unfilled layer flows downslope under the bed slope
``s`` with the flux ``K(W)\, s``, and thicker water carries more, so ``c = \mathrm d(K s)/\mathrm dW``. In laminar
flow ``K`` grows like ``W^3`` and ``c = 3|q|/W``, three times the mean velocity of the water ``|q|/W``, so a pulse of
water travels downslope faster than the water itself. Since thicker water moves faster, the back of a pulse
catches up with its front and the front steepens, while the diffusion from ``\partial W/\partial x`` spreads it
again.

To see what the wave means for the time step, take a row of cells in which each cell drains to its downstream
neighbour, with a flux ``c\, W`` out of every cell [LeVeque2002](@cite). Over a step the water balance of cell
``i``, with ``i-1`` its upstream neighbour, is

```math
W_{i}^{(m+1)} = W_{i}^{(m)} - r\, \big(W_{i} - W_{i-1}\big), \qquad r = \frac{c\, \Delta t}{\Delta x} ,
```

where ``r`` is the distance the wave travels in one step, measured in cells. A scheme that is explicit in the wave
takes the outflow and the inflow at the start of the step, so

```math
W_{i}^{(m+1)} = (1 - r)\, W_{i}^{(m)} + r\, W_{i-1}^{(m)} .
```

The cell keeps the fraction ``1 - r`` of its water and receives the fraction ``r`` of its neighbour's. For ``r > 1``
the cell sends out more water than it holds, its water thickness turns negative, and the swings grow from step to
step. The explicit scheme therefore needs ``r \le 1``, that is ``\Delta t \le \Delta x / c``, so the wave may cross at
most one cell per step. A scheme that is implicit in the wave takes the outflow and the inflow at the end of the
step, so

```math
W_{i}^{(m+1)} = \frac{W_{i}^{(m)} + r\, W_{i-1}^{(m+1)}}{1 + r} .
```

The new water thickness is a weighted average of the cell's old value and its upstream neighbour's new value, with
positive weights that add to 1. It cannot leave the range of these two values, so nothing grows at any time step and a positive water
thickness stays positive.
The cost is that each step mixes a cell with its upstream neighbour, which spreads a sharp front over several
cells, more so at large ``r``. Shakti.jl offers both ([Why the water thickness is differentiated](@ref WhyDifferentiate)).

### [Why the water thickness is differentiated](@id WhyDifferentiate)

Under `"implicit_upwind"` a face takes the water thickness of its donor at the end of the step, and the row
differentiates that water thickness in the donor's head ([Newton terms of the donors](@ref DonorNewtonTerms)).
This section shows why both are needed.

In an unfilled cell the water pressure is zero, so the hydraulic head is ``\hat{h} = z_b + W``.
In one dimension, with the bed slope ``s = -\partial z_b / \partial x``, the flux is
``q = K(W)\, s - K(W)\, \partial W / \partial x``, and the water balance becomes

```math
\frac{\partial W}{\partial t} + \frac{\partial}{\partial x}\big(K(W)\, s\big)
  = \frac{\partial}{\partial x}\left(K(W)\, \frac{\partial W}{\partial x}\right) + \frac{\dot m}{\rho_w} + i_{e\to b} .
```

The second term on the left is a kinematic wave ([Kinematic waves](@ref KinematicWave)), which carries the water
downslope at the speed ``c = \mathrm d(K s)/\mathrm dW``. The flux grows like ``W^3`` in laminar and like ``W^{3/2}`` in fully
turbulent flow, so ``c`` lies between ``\tfrac32 |q|/W`` and ``3|q|/W``. The right-hand side adds a nonlinear
diffusion. In an uplifted cell the hydraulic head is the overburden head plus the water thickness,
``\hat{h} = h_o + W``, and the same holds with the slope of ``h_o`` in place of the bed slope.

With `"upwind"` the face takes the donor's water thickness at the start of the step, so the scheme is explicit in
the wave, and its time step is limited to ``\Delta t \lesssim \Delta x / c`` ([Kinematic waves](@ref KinematicWave)).
With ``c = 3|q|/W`` in laminar flow this is

```math
\Delta t \lesssim \frac{\Delta x\, W}{3 |q|} .
```

For a layer of 5 mm carrying ``10^{-4}`` m² s⁻¹ at ``\Delta x = 100`` m this is about half an hour. The limit
comes with the unfilled and uplifted regimes. On the filled sheet the water thickness is the gap, which changes
only through the gap update, so the filled sheet does not have it. Shakti.jl warns when `"upwind"` is chosen with
unfilled cavities or uplift.

`"implicit_upwind"` removes the limit. It builds the transmissivity of each face from its donor's water thickness at
the end of the step, ``W_d = \mathcal{W}_d(h_d)``, which depends on the unknown head of the donor. With the Newton terms ``J_f``, the step is the implicit scheme
of [Kinematic waves](@ref KinematicWave), so the water thickness does not grow at any time step. The cost is that
sharp fronts are spread over several cells.

Taking the water thickness at the end of the step without its derivative does not remove the limit. Consider one
unfilled cell with an inflow ``Q`` per unit area, draining through one face to a neighbour of fixed hydraulic head
``\hat{h}_F`` that lies ``\Delta z = z_b - \hat{h}_F \ge 0`` below its bed. If each linear solve takes the
transmissivity at the water thickness of the previous iterate, without its derivative, the row and the gradient of
the map from one iterate to the next are

```math
\frac{W^{(k+1)} - W^{(m)}}{\Delta t} + \frac{K(W^{(k)})}{\Delta x^2}\left(\Delta z + W^{(k+1)}\right) = Q, \qquad
\frac{\mathrm dW^{(k+1)}}{\mathrm dW^{(k)}} = -\frac{(\mathrm dK/\mathrm dW)(W^{(k)})\,(\Delta z + W^{(k+1)})/\Delta x^2}{1/\Delta t + K(W^{(k)})/\Delta x^2} .
```

The iteration converges only where this gradient is smaller than 1 in magnitude. At small ``\Delta t`` it is
about ``-r``, which represents the (negative of the) distance the wave travels in one step in cells ([Kinematic waves](@ref KinematicWave)), so the iteration converges only below the explicit limit. At large ``\Delta t`` it
tends to ``-(\mathrm dK/\mathrm dW)(\Delta z + W)/K``, which is ``-3(W + \Delta z)/W`` in laminar flow. Its magnitude
is at least 3, so each iterate moves further from the solution. A Newton step includes ``\mathrm dK/\mathrm dW``, and
its map has gradient zero at the solution, so near the solution it converges at any ``\Delta t``.

## [Gap update](@id GapUpdate)

`FullyImplicitGapScheme` evaluates both the opening and the closure at ``b^{(m+1)}``:

```math
b^{(m+1)} = b^{(m)} + \Delta t \left( \frac{\dot m}{\rho_i} + \gamma \left(\mathcal{B} - b^{(m+1)}\right)_+ - C\, b^{(m+1)} \right),
\qquad \gamma = \frac{|\mathbf{u}_b|}{l_r}, \quad C = A |N|^{n-1} N ,
```

at ``N = N^{(m+1)}`` and ``\dot m = \dot m^{(m+1)}``, with ``\mathcal{B}`` the bump height of
[Gap height](@ref). The equation is piecewise linear in ``b^{(m+1)}``, so it is solved in closed form:

```math
b^{(m+1)} =
\begin{cases}
\dfrac{b^{(m)} + \Delta t\, (\dot m / \rho_i + \gamma \mathcal{B})}{1 + \Delta t\, (\gamma + C)} & \text{if this is } < \mathcal{B} , \\[2ex]
\max\!\left(\dfrac{b^{(m)} + \Delta t\, \dot m / \rho_i}{1 + \Delta t\, C},\; \mathcal{B}\right) & \text{otherwise.}
\end{cases}
```

The result is clamped to ``[b_{\min}, b_{\max}]``, and the update holds for any ``\Delta t``.

`ExplicitGapScheme` is forward Euler, ``b^{(m+1)} = b^{(m)} + \Delta t\, \dot b(N^{(m+1)}, b^{(m)})``, accurate and free
of overshoot only for ``\Delta t\, (\gamma + C) < 1``.

On the filled sheet without uplift the water thickness is the gap, ``W = b``, and the gap cannot close below
``b_{\min}``. A cell that its fluxes drain would therefore keep ``b_{\min}`` of water whatever its head. Its
water-thickness function would have zero gradient with respect to ``h`` at the clamp, so its row would lose the
term ``(\mathrm d\mathcal{W}/\mathrm dh)/\Delta t`` from its diagonal, and the cell would keep its transmissivity ``K(b_{\min})`` and pass on its neighbours' water at an almost
flat head. The results would then depend on the choice of ``b_{\min}``. Under `ConsistentGapCoupling` without gap diffusion,
Shakti.jl therefore lets the water drain below the gap. The new water thickness ``W^{(m+1)}`` is the gap update of
the chosen scheme, the closed form of `FullyImplicitGapScheme` or the forward Euler step of `ExplicitGapScheme`,
with ``W^{(m)}`` in place of ``b^{(m)}`` and clamped to ``[0, b_{\max}]``, and the gap is
``b^{(m+1)} = \max(W^{(m+1)}, b_{\min})``. Where ``W^{(m+1)} \ge b_{\min}`` this is the ordinary filled sheet,
``W = b``. Where ``W^{(m+1)} < b_{\min}`` the gap holds ``b_{\min} - W^{(m+1)}`` of empty space, the dry floor. The
filled regime of the head row and the transmissivity of each face take the same ``W^{(m+1)}``, so the row and the gap
update agree on the water, and a drained cell carries no more water. The results then no longer depend on ``b_{\min}``.

### [Accuracy of the gap update](@id GapAccuracy)

Each case of the closed form, below the bump and above it, is a backward-Euler step of a linear equation, that
is, a step that takes the rate at the end of the step, at ``b^{(m+1)}``. The equation is

```math
\dot b = S - R\, b ,
```

with ``S = \dot m/\rho_i + \gamma \mathcal{B}`` and ``R = \gamma + C`` below the bump, and ``S = \dot m/\rho_i`` and
``R = C`` above it. Both are held fixed over the step. With them fixed, the gap would settle at
``b^* = S/R``, the gap at which opening and closure balance. Write ``e = b - b^*`` for the distance of the
gap from that value. Since ``S = R\, b^*``, the equation becomes ``\dot e = -R\, e``, so the distance decays
exponentially at the rate ``R``, and ``1/R`` is the relaxation time of the cell.

Over one step, both the exact solution and the update multiply the distance by a factor. The exact
solution multiplies it by ``\exp(-R\Delta t)``, and the update by ``1/(1 + R\Delta t)``.

```math
e^{(m+1)} = \exp(-R\Delta t)\, e^{(m)} \quad \text{(exact)}, \qquad
e^{(m+1)} = \frac{e^{(m)}}{1 + R\Delta t} \quad \text{(update)} .
```

The second follows from the backward-Euler step ``(e^{(m+1)} - e^{(m)})/\Delta t = -R\, e^{(m+1)}``, or by
subtracting ``b^*`` from both sides of the closed form above. Forward Euler (`ExplicitGapScheme`) gives
``e^{(m+1)} = (1 - R\Delta t)\, e^{(m)}`` instead, which changes sign for ``R\Delta t > 1`` (the gap overshoots
``b^*``) and grows for ``R\Delta t > 2`` (the update diverges).

For small ``R\Delta t`` the two factors agree in their first two terms, ``1 - R\Delta t``, and differ by about
``\tfrac12 (R\Delta t)^2``. Each step therefore incurs an error of about ``\tfrac12 (R\Delta t)^2 e^{(m)}``, and over a run
of length ``T`` the ``T/\Delta t`` steps add up to an error proportional to ``\Delta t`` and to ``R^2``. The update is first-order
accurate in time, like forward Euler.

For large ``R\Delta t`` both factors are close to zero, so the exact solution and the update both put the
gap close to ``b^*``. The error lies in how fast the gap
approaches ``b^*``, not in the value it approaches. Where the ice is thick and the water pressure is far below
overburden, ``N`` is of order 10 MPa and ``1/R`` is of order minutes. There the true gap reaches ``b^*`` within minutes, so at a time step of hours the update and the
true gap agree.

The analysis holds for ``R \ge 0``. For ``R < 0`` the distance from ``b^*`` grows instead of decaying, and
Shakti.jl replaces the update by the exact solution over the step ([Negative effective pressure](@ref NegativeN)).

### [Derivative of the gap update](@id GapDerivative)

In a filled cell the water thickness at the end of the step is the gap the update leaves, or, on the dry floor, the
water the update leaves, ``\mathcal{W}_P(h) = \mathcal{G}_P(N)`` with ``N = p_o - \rho_w g (h - z_b)``. The head
equation linearizes this water thickness about the iterate ([Picard iteration](@ref PicardIteration)), so it needs
the gradient

```math
\frac{\mathrm d \mathcal{G}_P}{\mathrm d h} = -\rho_w g\, \frac{\mathrm d \mathcal{G}_P}{\mathrm d N} .
```

Where the gap is not at ``b_{\mathrm{min}}`` or ``b_{\mathrm{max}}``, differentiating the backward-Euler update
``b^{(m+1)} = (b^{(m)} + \Delta t\, S)/(1 + \Delta t\, R)`` through the creep rate ``C = A |N|^{n-1} N``, which
enters ``R``, gives

```math
\frac{\mathrm d \mathcal{G}_P}{\mathrm d N} = -\frac{n A |N|^{n-1}\, \Delta t\, \mathcal{G}_P}{1 + \Delta t\, R} \le 0 .
```

A larger effective pressure closes the gap faster, so the update leaves a smaller gap. Where the gap is at
``b_{\mathrm{min}}`` or ``b_{\mathrm{max}}`` it does not change with ``N``, and the gradient is zero.

The row puts this gradient divided by ``\Delta t`` on its diagonal, and two limits show what it means. For
``\Delta t\, R \ll 1``, ``\Delta t^{-1}\, \mathrm d \mathcal{G}_P / \mathrm d h`` tends to
``\rho_w g\, n A |N|^{n-1} b^{(m)}``, the Newton term of the creep closure in the row of `ExplicitGapCoupling`
([Gap couplings](@ref GapCouplings)). For ``\Delta t\, R \gg 1`` and no sliding,
``\mathrm d \mathcal{G}_P / \mathrm d h \approx \rho_w g\, n\, \mathcal{G}_P / N``, so the diagonal term vanishes as
``\Delta t`` grows. Where the ice is thick and ``N`` is large, the gap reaches its balance value within a step, so
``\mathcal{G}_P`` and its gradient are small, and the diagonal of a filled cell is small. This is one reason the
iteration limits its steps ([Limiting the head iterate](@ref LimitingTheHead)).

Shakti.jl does not code this formula. It evaluates the gradient by a central difference (`scheme_head_terms`),

```math
\frac{\mathrm d \mathcal{G}_P}{\mathrm d N} \approx \frac{\mathcal{G}_P(N + \delta) - \mathcal{G}_P(N - \delta)}{2\delta},
\qquad \delta = 10^{-6} \max\!\left(|N|,\, 10^{3}\ \mathrm{Pa}\right),
```

and sets a positive value of it to zero, so that ``\mathrm d \mathcal{G}_P / \mathrm d h \ge 0``. One formula then
covers both cases of the update, the exponential step at negative effective pressure
([Negative effective pressure](@ref NegativeN)), the clamps, the creep cutoff and the dry floor of the filled
sheet, each of which would otherwise need its own derivative. The floor of ``10^3`` Pa in ``\delta`` keeps the
step from vanishing at ``N = 0``.

### [Negative effective pressure](@id NegativeN)

Each case of the update is a step of the linear equation ``\dot b = S - R\, b``
([Accuracy of the gap update](@ref GapAccuracy)). Where ``N < 0`` the water pressure exceeds the weight of
the ice, the creep rate ``C`` is negative, and creep opens the gap instead of closing it. The rate ``R`` can
then be negative. Above the bump ``R = C < 0`` whenever ``N < 0``, and below it ``R = \gamma + C < 0`` where
sliding is slow. With ``R < 0`` the distance of the gap from ``b^*`` grows over the step instead of decaying.
Backward Euler would divide by ``1 - \Delta t\, |R|``. That denominator vanishes at ``\Delta t = 1/|R|``, where
the update blows up, and is negative beyond it, where the update gives a negative gap.

For ``R < 0`` Shakti.jl therefore integrates that case exactly over the step, with ``S`` and ``R`` held
fixed. Writing ``R = -|R|``, the equation is ``\dot b = S + |R|\, b``. Multiplying it by ``\exp(-|R|\, t)`` gives

```math
\frac{d}{dt}\Big(b\, \exp(-|R|\, t)\Big) = S\, \exp(-|R|\, t) .
```

Integrating over the step, from ``b(t^{(m)}) = b^{(m)}`` to ``b(t^{(m+1)}) = b^{(m+1)}``, and multiplying by
``\exp(|R|\, t^{(m+1)})`` leaves only the step length ``\Delta t = t^{(m+1)} - t^{(m)}``,

```math
b^{(m+1)} = b^{(m)} \exp(x) + S\, \Delta t\, \frac{\exp(x) - 1}{x}, \qquad x = |R|\, \Delta t .
```

The code evaluates ``\exp(x) - 1`` with the library function `expm1`. For small ``x``, ``\exp(x) - 1``
subtracts two nearly equal numbers and loses digits, which `expm1` avoids. As ``R \to 0^-``,
``(\exp(x) - 1)/x \to 1`` and the update tends to ``b^{(m)} + S\, \Delta t``, the backward-Euler value at
``R = 0``. The update is therefore continuous in ``R``, and it stays finite for any time step that does not
overflow the exponential (``x \gtrsim 700``). This is one reason the iteration limits how far ``N`` may fall
below zero in one iterate ([Limiting the head iterate](@ref LimitingTheHead)). The result is clamped to
``[b_{\min}, b_{\max}]`` as in the other cases.

The update for ``R \ge 0`` keeps backward Euler. The exponential step is exact only while ``S``, ``R`` and
the case stay fixed over the step. Backward Euler instead evaluates them at the gap at the end of the
step, so it picks the case the new gap lies in, and with the creep cutoff
([Creep cutoff](@ref CreepCutoff)) it also covers the parts of the range where the closure is quadratic in
``b``. Both schemes are first-order accurate in time, since ``N`` changes over the step anyway.

The floor ``\max(\cdot, \mathcal{B})`` in the second case of the update matters only at ``N < 0``. For ``C \ge 0``,
a first case at or above ``\mathcal{B}`` implies a second case at or above ``\mathcal{B}`` too, so the floor never
binds. For ``C < 0`` this no longer holds, and the exponential step of the second case can land below
``\mathcal{B}`` after the first case has been rejected. The update then returns ``\mathcal{B}``, which keeps it
continuous in ``b^{(m)}`` and ``N``.

### [Creep cutoff](@id CreepCutoff)

With ``b_c > 0`` the creep length is ``l_c = b^2/b_c`` for ``b \le b_c`` and ``b`` above ([Gap height](@ref)), so
the closure ``C\, l_c`` is quadratic in ``b`` below ``b_c``, and very thin gaps close more slowly, since
``l_c = b^2/b_c`` is smaller than ``b`` there. The rate of
the gap changes its formula at two gap values, the bump height ``\mathcal{B}``, above which sliding no longer opens
the gap, and ``b_c``, above which ``l_c = b``. These two thresholds divide the range of ``b`` into four parts, and
backward Euler gives one candidate gap on each. Above ``b_c`` the candidates are the solutions of the two cases of
[Gap update](@ref GapUpdate), below and above the bump. Below ``b_c`` they are the positive roots of

```math
\frac{\Delta t\, C}{b_c}\, b^2 + (1 + \Delta t\, \gamma_j)\, b - \left(b^{(m)} + \Delta t\, S_j\right) = 0, \qquad j = 1, 2,
```

with ``\gamma_1 = \gamma`` and ``S_1 = \dot m/\rho_i + \gamma \mathcal{B}`` below the bump, and ``\gamma_2 = 0`` and
``S_2 = \dot m/\rho_i`` above it. For ``C > 0`` the residual of the update, ``b - b^{(m)}`` minus ``\Delta t`` times the rate at ``b``,
increases with ``b``, so exactly one
candidate lies in the part of the range it was derived for. The code tests the candidates in turn. The solution can
lie exactly on a threshold. The candidates of the two parts that meet there then come out within rounding of the
threshold, and can each fall just on the wrong side of it, so that no candidate lies in its own part. The code
then returns the threshold, ``b_c`` or ``\mathcal{B}``, at which the residual is smaller.

Writing the quadratic as ``a_2 b^2 + a_1 b - a_0 = 0`` with ``a_2, a_0 > 0``, its positive root is computed as

```math
b = \frac{2 a_0}{a_1 + \sqrt{a_1^2 + 4 a_2 a_0}} ,
```

the same form as the root for the Reynolds number in [Face flux](@ref FaceFlux), which avoids subtracting two
nearly equal numbers when ``4 a_2 a_0 \ll a_1^2``, at small time steps and thin gaps.

For ``C < 0``, where creep opens the gap, the gap updates use ``l_c = b``. An opening rate ``|C|\, b^2/b_c`` grows
faster than the gap itself and would make the gap infinite in a finite time. The explicit-coupling rows and the
gap-diffusion system use ``l_c`` from the start of the step, and choose its form from the sign of ``N`` at the end
of the previous step.

### [Gap diffusion](@id GapDiffusion)

Under `WithDiffusion` the gap equation gains the wall-melt term ``\nabla\cdot(\mathcal{D}\nabla b)``
([Gap height](@ref)), which couples neighbouring cells. No cell can then compute its gap alone, and the gap
update becomes a second linear system, that of SUHMO [Felden2023](@cite),

```math
\left(1 + \sum_f d_f + \delta_P\right) b^{(m+1)}_P - \sum_{f:\,F\ \text{grounded}} d_f\, b^{(m+1)}_F
  = b^{(m)}_P + \Delta t\, s_{b,P}, \qquad d_f = \frac{\Delta t\, \mathcal{D}_f}{\Delta_f^2},
```

with

```math
\delta_P = \Delta t\, (C)_+ \frac{l_c}{b^{(m)}} + \Delta t\, \gamma_\beta, \qquad
s_{b,P} = \frac{\dot m}{\rho_i} + (-C)_+\, l_c + \gamma_\beta\, \mathcal{B},
```

where ``(x)_+ = \max(x, 0)``, and ``\gamma_\beta = \gamma`` where the lagged ``\beta`` is positive and 0 elsewhere.
Closure is implicit for ``C > 0``, through the lagged ratio ``l_c / b^{(m)}`` (exact for ``l_c = b``), and
explicit for ``C \le 0``. Opening by sliding is implicit where the lagged ``\beta`` is positive, and the melt is
explicit. SUHMO instead takes the closure and the opening by sliding at the start-of-step gap, on the right-hand
side, and only the diffusion at the new gap. Both terms are linear in the new gap and add only positive entries to
the diagonal, as in `FullyImplicitGapScheme`, so taking them at the new gap keeps the properties of the matrix
below. Faces to cells that are not grounded carry no diffusion. The matrix is symmetric, with a diagonal of
at least ``1 + \sum_f d_f``, so its eigenvalues are at least 1. It is positive definite and well conditioned,
and it takes its own solver ([Linear solvers](@ref LinearSolvers)). The result is clamped to
``[b_{\min}, b_{\max}]``.

The diffusion coefficient ``\mathcal{D}`` is built per cell from the dissipation and the sensible heat of
[Melt on the grid](@ref MeltOnTheGrid),

```math
\mathcal{D}_P = \max\!\left(0,\; \frac{\min(\max(W_P, 0),\, b_P)}{\rho_i L} \left( Q_{\mathrm{diss},P} + Q_{\mathrm{sens},P} \right)\right), \qquad
\mathcal{D}_f = \max\!\left(\tfrac12 (\mathcal{D}_P + \mathcal{D}_F),\; \mathcal{D}_{\min}\right) .
```

``\min(W, b)`` is the height of the gap walls that the water wets, the whole gap in a filled cell and only the
water thickness in an unfilled one. In an uplifted cell it is the gap ``b``. The water above the gap is the
uplift layer, a sheet under the lifted ice that spans the whole cell, so it adds no side wall to melt. The floor at zero matters on an adverse bed slope, where the sensible heat is
negative. A negative ``\mathcal{D}`` would be backward diffusion, which has no stable solution. The cell values
are averaged to the faces, which take at least `D_min` (m² s⁻¹, default 0), set as
`WithDiffusion(ls; D_min, D_faces)`. Averaging cell values lets a channel that runs along one grid direction also diffuse
sideways, across it. A coefficient built per face from that face's own flux would take the small flux on the faces
beside the channel instead of the channel's heat, so the channel would hardly widen. A diagonal channel crosses
both kinds of face and would widen, so the result would depend on the orientation of the grid. With `D_faces = true` the face coefficient is instead the mean wet gap of the
two cells times the mean of their heats, as in SUHMO,

```math
\mathcal{D}_f = \max\!\left(\frac{w_P + w_F}{2}\, \frac{H_P^+ + H_F^+}{2}\, \frac{1}{\rho_i L},\; \mathcal{D}_{\min}\right),
\qquad w = \min(\max(W, 0),\, b), \quad H = Q_{\mathrm{diss}} + Q_{\mathrm{sens}}, \quad H^+ = \max(H, 0) .
```

The two forms differ only in the order of averaging and multiplying. At the face between a channel cell ``P`` and
a dry neighbour ``F``, with ``w_F = 0`` and ``H_F = 0``, the default gives ``\mathcal{D}_f = \mathcal{D}_P/2`` and
`D_faces = true` gives ``\mathcal{D}_f = \mathcal{D}_P/4``, so SUHMO's form spreads a channel into dry ground
half as fast.

Under `ConsistentGapCoupling` each head row stores the water thickness its cell will hold at the end of the step,
which in a filled cell is the new gap ([Gap couplings](@ref GapCouplings)). Without diffusion the new gap of a
cell depends only on its own effective pressure, ``\mathcal{G}_P(N)``, so the row can compute it from its own
head. With diffusion it also depends on the new gaps of the neighbours, through the term ``d_f\, b_F^{(m+1)}`` of
the gap system, and theirs depend on their own neighbours. The new gap of one cell then depends on the heads of
all cells, and only a solve of the whole gap system gives it. Shakti.jl therefore solves the gap system inside
the Picard iteration, in the following steps.

1. After each head solve, the gap system is solved at the new ``N``, ``\dot m`` and ``\mathcal{D}``. This gives the
   new gap of every cell, ``b_{\mathrm{next}}``.
2. The next head row of each cell stores ``b_{\mathrm{next}}`` as the water thickness at the end of the step, so the
   water the cell gains over the step is ``b_{\mathrm{next}} - W^{(m)}``. In an unfilled or uplifted cell it adds
   ``h - z_b`` or ``h - h_o`` as in [Gap couplings](@ref GapCouplings).
3. The row also needs the gradient of ``b_{\mathrm{next}}`` with respect to the cell's head, its Newton term. The
   exact gradient would include how the neighbours' new gaps change with this cell's head, which would need a
   further solve. Shakti.jl instead takes the cell's own row of the gap system,
   ``b_P = (b^{(m)}_P + \Delta t\, s_{b,P} + \sum_f d_f\, b_F)/(1 + \sum_f d_f + \delta_P)``, holds the neighbours'
   gaps ``b_F`` at ``b_{\mathrm{next}}``, and differentiates it in ``N`` by the central difference of
   [Derivative of the gap update](@ref GapDerivative). The Newton term then keeps the cell's dependence on its own
   head and drops the dependence through its neighbours. Like the other held terms of the Picard iteration, this
   changes how fast the iteration converges, not the head it converges to, since the converged row stores the
   ``b_{\mathrm{next}}`` of the whole system.
4. Under `"implicit_upwind"` faces (the default), the transmissivity of each face is built from its donor's water
   thickness at the end of the step, so it needs the new gap too. The fluxes are computed from the transmissivity,
   and ``\dot m`` and ``\mathcal{D}`` from the fluxes, so this gap cannot wait for them. After each head solve the
   gap system is therefore solved twice. The first solve uses the new ``N`` with the ``\dot m`` and
   ``\mathcal{D}`` of the previous iterate, and gives the gap for the transmissivity. The fluxes, ``\dot m`` and
   ``\mathcal{D}`` are then updated, and the second solve gives the gap for the next head rows (step 1). The
   transmissivity takes the smaller of the start-of-step gap ``b^{(m)}`` and the gap of the first solve. Where the gap
   closes, it takes the new gap, so a closing cell does not pass on water it no longer holds. Where the gap opens,
   it keeps ``b^{(m)}``, so the opening reaches the transmissivity only at the next step. The melt in the transmissivity
   lags in the same way ([Melt in the transmissivity](@ref MeltInTransmissivity)), which keeps a growing gap from
   raising the flux, the melt and the gap again within one step.

The same solves are made once before the first iterate, at the extrapolated head. A step of ``K`` iterates
therefore solves the gap system ``K + 1`` times, and ``2K + 2`` times under `"implicit_upwind"`. The last solve,
at the converged head, is the gap update of the step, ``b^{(m+1)} = b_{\mathrm{next}}``, so the head rows and the
gap update agree on the water.

Under `ExplicitGapCoupling` the head rows take the gap change from the rate of
[Gap couplings](@ref GapCouplings), evaluated at the start-of-step gap, and add
``-\nabla\cdot(\mathcal{D}\nabla b^{(m)})``, the diffusion of the start-of-step gap, to the right-hand side. The gap
system then moves the gap with the closure, the opening by sliding and the diffusion of the end-of-step gap. The
gap the rows assume and the gap the system returns therefore differ, and the water between them is created or
lost. The difference grows with ``\Delta t``, so this coupling does not conserve water at long time steps. This
is the coupling of SUHMO, which solves the head with the gap held at its start-of-step value and then solves the
gap system once at the new head [Felden2023](@cite). It costs one gap solve per step, against ``K + 1`` or
``2K + 2`` under `ConsistentGapCoupling`, and suits short time steps or reproducing SUHMO.

## [Water and the budget](@id WaterAndBudget)

Under `UnfilledCavities` the water thickness is advanced with the converged fluxes,

```math
W^{(m+1)}_P = \max\!\left(0,\; W^{(m)}_P + \Delta t \left( \frac{\dot m_P}{\rho_w} + i_{e\to b,P} - (\nabla\cdot\mathbf{q})_P \right) - \left( b_e(h^{(m+1)}_P) - b^{(m)}_{e,P} \right) \right),
```

and ``b^{(m+1)}_e = b_e(h^{(m+1)})`` is stored for the next step. The
gap is updated separately ([Gap update](@ref GapUpdate)), so the empty part of the gap is ``b^{(m+1)} - W^{(m+1)}``,
and any water above the gap is the uplift layer. On the other sheets the water thickness is not advanced from the
budget. On the dry floor (`FilledCavities`, `NoUplift` and `ConsistentGapCoupling` without gap diffusion) the gap
update sets ``W^{(m+1)}`` ([Gap update](@ref GapUpdate)). Otherwise the water thickness is the new gap, plus the
uplift layer under `IceUplift`, ``W^{(m+1)} = b^{(m+1)} + (h^{(m+1)} - h_o)_+``. Under `NoUplift` it is the gap alone,
``W^{(m+1)} = b^{(m+1)}``, even where ``h^{(m+1)} > h_o``. Under `ConsistentGapCoupling` this is
``\mathcal{W}_P(h^{(m+1)})``.

## [Conservation](@id Conservation)

The two cells of a face read the same flux ``q_f``. Both rows use the same ``K_f`` and the same difference of
``\hat{h}``, so the water that leaves one cell through the face enters the other. Summed over any set of cells,
the fluxes of its interior faces cancel, and the water the set holds changes only by its sources and by the flux
through its outer faces. This holds at every head, not only at convergence, so the fluxes never create or lose
water. For the same reason ``K_f`` is computed once per face and iterate and the same value is used in the matrix,
the melt rate and the budget below.

Summed over all grounded cells, the water under grounded ice, the total water thickness times the cell area,
``V^{(m)} = \sum_P \left(W^{(m)}_P + b^{(m)}_{e,P}\right) \Delta x \Delta y``, changes over a step as

```math
V^{(m+1)} - V^{(m)} = \Delta t \left( I - O \right) + V_{\mathrm{created}} ,
\qquad
I = \sum_P \left( \frac{\dot m_P}{\rho_w} + i_{e\to b,P} \right) \Delta x \Delta y ,
\qquad
O = \sum_{f \to \text{land, ocean}} q_f \, \Delta s_f ,
```

where ``O`` sums the outflow through the faces to `LAND` and `OCEAN` cells and ``\Delta s_f`` is the length of
the face (``\Delta y`` for an ``x`` face). Faces to `OTHER_BASIN`, `FROZEN_BED` and the edge of the grid carry no
flux. ``V_{\mathrm{created}}`` is the water a step leaves beyond its sources and fluxes. It has three sources.

1. Under `UnfilledCavities`, the floor ``W \ge 0`` of the update above. A cell whose fluxes take out more water
   than it holds has its water thickness set to zero, which creates the missing water. With `"upwind"` faces this
   happens above the time-step limit of the rule, and with `"implicit_upwind"` faces only in unconverged steps
   ([Why the water thickness is differentiated](@ref WhyDifferentiate)). The averaging rules (`"arithmetic"`, `"harmonic"`, `"gap_mean"`) would do so at
   any time step, which is why they are refused under `UnfilledCavities`. Where the head rows and the gap update
   disagree on the water, as under `ExplicitGapCoupling` or in an unconverged step, the difference stays in
   ``W - b``, as empty gap or as water above the gap, and creates water only where it drives ``W`` below zero.
2. On the other sheets, the coupling. Their water thickness is taken from the head and the gap, not advanced
   with the fluxes, so it differs from the budget wherever the head rows and the gap update disagree on the water.
   Under `ExplicitGapCoupling` they disagree wherever creep is fast or a clamp at ``b_{\min}`` or ``b_{\max}``
   binds ([Gap couplings](@ref GapCouplings)). Under `ConsistentGapCoupling` they agree once the step converges.
3. On the other sheets, an unconverged step. Its water thickness keeps the residual of its rows.

`water_created(sim)` returns ``V_{\mathrm{created}}`` of the last step and its sums over all steps, also summed by
sign (`gross_created`, `gross_lost`) so that water created in one cell or step and lost in another does not cancel.
Under `ConsistentGapCoupling` a converged step creates water of the order of the Picard tolerance times the input,
and under `ExplicitGapCoupling` percents of the input or more.

Where the water test applies ([Convergence tests](@ref ConvergenceTests)), it also bounds what a converged step can
create. Each cell is balanced to ``\epsilon_w \Sigma_P``, so

```math
\left| V_{\mathrm{created}} \right| \lesssim \epsilon_w \sum_P \Sigma_P \, \Delta x \Delta y ,
```

with ``\Sigma_P`` the cell's water in the gap at the start and the end of the step, the change of its englacial
water, and the magnitudes of its melt (or freeze-on) and input from above over the step. The englacial water
itself and the face fluxes do not enter. This is a worst case, far above the water converged steps create in
practice. But it is set by the water in the gap as well as by the sources, so where the gap holds much more water
than the sources bring in one step, the bound allows a created water that is a sizeable fraction of that step's
input. Water removed by mask changes is not part of
it ([Mask changes](@ref MaskChanges)).

## [Melt on the grid](@id MeltOnTheGrid)

The melt rate of a cell is

```math
\dot m_P = \max\!\left( \frac{Q_{G,P} + Q_{b,P} + Q_{\mathrm{diss},P} + Q_{\mathrm{sens},P} - Q_{T,P}}{L},\; \dot m_{\min,P} \right),
```

with the heat terms of [Melt rate](@ref) and, with `limit_freeze_on` (the default), the floor
``\dot m_{\min} = -\rho_i C_{\mathrm{cap}}`` of [Freeze-on capacity](@ref FreezeOnCapacity). Each heat term can be
switched off, and it is then not computed at all. With `gap_opening_melt = :exclude_uniform` the gap opens by less
than ``\dot m/\rho_i``, since part of the uniform melt makes water but no gap, and the floor then applies to the
part that opens the gap ([Melt in the gap opening](@ref GapOpeningMelt)).

The floor keeps a freezing cell from freezing its gap shut below ``b_{\min}``, or from freezing more water than it
holds, within one step. It is computed once per step from the start-of-step state, again at ``\Delta t/2`` when
the adaptive time step retries a step, and held during the iteration, so the head rows see a fixed limit.

The heat terms are built from face quantities, the flux and the gradients, which the rest of this section turns
into cell values.

The dissipation and the sensible heat are built from face products. On the staggered grid both factors of ``\mathbf{q}\cdot\nabla\hat{h}`` and of
``\mathbf{q}\cdot\nabla p_w`` live on the faces. On face ``f``, ``q_f (\nabla\hat{h})_f`` is the power dissipated
per unit bed area in a strip of width ``\Delta_f`` centred on the face. That strip lies half in each of the two
cells, so each cell receives half of it,

```math
Q_{\mathrm{diss},P} = \rho_w g \left| \sum_f s_f\, q_f (\nabla\hat{h})_f \right|, \qquad
Q_{\mathrm{sens},P} = c_t c_w \rho_w \sum_f s_f\, q_f (\nabla p_w)_f ,
```

with ``s_f = \tfrac12`` on a face between two grounded cells. A face to an `OCEAN` or `LAND` cell has
``s_f = 1``. The boundary cell has no hydrology to receive its half of the heat of the outflow, so the grounded
cell takes all of it. For uniform ``\mathbf{q}`` and ``\nabla\hat{h}`` the two ``x`` faces of a cell contribute
``q_x \partial_x\hat{h}`` and the two ``y`` faces ``q_y \partial_y\hat{h}``, so the sum gives
``\mathbf{q}\cdot\nabla\hat{h}`` exactly. Summed over all cells, the power of every face is counted once, so no
heat is lost or created. Since ``q_f = -K_f (\nabla\hat{h})_f``, every face product
``q_f (\nabla\hat{h})_f = -K_f (\nabla\hat{h})_f^2`` is negative or zero, and the absolute value only turns the
sum into a positive heat.

The frictional heat ``Q_b = |\mathbf{u}_b\cdot\boldsymbol{\tau}_b|`` averages the face values of the sliding
velocity times the shear stress, with weight ``\tfrac12`` on every face, faces to land and ocean included,

```math
Q_{b,P} = \left| \tfrac12 (u_x\tau_x)_{i+\frac12} + \tfrac12 (u_x\tau_x)_{i-\frac12}
  + \tfrac12 (u_y\tau_y)_{j+\frac12} + \tfrac12 (u_y\tau_y)_{j-\frac12} \right| .
```

Friction, unlike dissipation, is made on the bed of the cell itself, not along the path between two cell
centres. Beyond a face to land or ocean there is no ice-covered bed whose heat the grounded cell would have to
take over. With uniform sliding the average gives ``\mathbf{u}_b\cdot\boldsymbol{\tau}_b`` exactly. Giving the
grounded cell the full value of a face to land or ocean, as for the dissipation, would count that face twice,
and give 1.5 times the true heat for sliding normal to it. The shear stress on a face is computed from the face
value of ``N``, the mean of its two cells, or the grounded cell's own value at a boundary face, floored at zero.
