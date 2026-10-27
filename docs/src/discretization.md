# Discretization

This page shows how the equations of [Model equations](@ref) are discretized. Space is discretized
by finite volumes on a regular grid and time by backward Euler. Each step solves the head equation
by a Picard iteration and then moves the gap. Unless said otherwise the defaults are described:
the GlaDS-2 sheet, `ConsistentGapCoupling`, `"implicit_upwind"` faces and `FullyImplicitGapScheme`.

## Grid

`Grid(nx, ny, lx, ly)` has ``n_x \times n_y`` cells with centres
``\mathbf{x}_{ij} = ((i-1)\Delta x, (j-1)\Delta y)``, ``\Delta x = l_x / (n_x - 1)``,
``\Delta y = l_y / (n_y - 1)``. The fields ``b``, ``W``, ``h``, ``p_w``, ``N`` and ``\dot m`` live at cell
centres; ``K``, ``\mathrm{Re}`` and the flux ``q`` live on faces (staggered, `q_x` of size
``(n_x + 1) \times n_y`` and `q_y` of size ``n_x \times (n_y + 1)``). The face formulas are written for
the ``x`` face ``i - \tfrac12`` between cells ``i - 1`` and ``i``; the ``y`` faces are the same with ``\Delta y``.

## Time step

Step ``n \to n+1`` of length ``\Delta t`` (`step!`):

1. From the start-of-step gap ``b^n`` and water ``W^n``: the overburden ``p_o``, the jumps of the gap
   across faces (for ``\phi = h + b``), the opening coefficient ``\beta`` and the freeze-on limit.
   These are held for the whole step. A coupled ice model may have changed ``z_s``, ``z_b``,
   ``\mathbf{u}_b`` before this.
2. Solve the head equation for ``h^{n+1}`` by Picard iteration. Every field that depends on the head,
   ``p_w``, ``N``, ``\mathbf{q}``, ``\mathrm{Re}``, ``\boldsymbol{\tau}_b`` and ``\dot m``, is recomputed
   at every iterate, so at convergence they are evaluated at ``n+1``.
3. Move the gap, ``b^{n+1} = \mathcal{G}(b^n; N^{n+1}, \dot m^{n+1})``, with the gap scheme ``\mathcal{G}``.
4. Set the water ``W^{n+1}`` and refresh ``\beta`` and ``l_c`` from ``b^{n+1}`` for the next step.

The head equation contains ``\partial b / \partial t``, which step 3 has not computed yet. The gap
coupling decides what step 2 assumes for it. Under `ConsistentGapCoupling` the head rows assume
exactly ``\mathcal{G}``, so steps 2 and 3 store the same water and the water budget closes at any
``\Delta t``. Under `ExplicitGapCoupling` (Shakti) they assume
``b^{n+1} - b^n = \Delta t\, \dot b(N^{n+1}, b^n)``, which differs from ``\mathcal{G}`` where creep is
fast (``\Delta t A |N|^n > 1``), and the difference is created water ([Gap coupling](@ref GapCoupling)).

## Face flux

On a face the driving gradient is the difference of the potential ``\phi = h + b``, with the gap held at
``b^n``:

```math
(\nabla\phi)_{i-\frac12} = \frac{(h_i + b^n_i) - (h_{i-1} + b^n_{i-1})}{\Delta x} ,
```

and without the ``b^n`` terms under `BedPotential`. At a face to an `OCEAN` or `LAND` cell the boundary side's gap counts as 0, since its fixed head is
already the level of its water. The laminar conductance of the face is

```math
K_{0,f} = \frac{g}{12\nu} \, \overline{W^3}_f ,
```

where ``\overline{W^3}_f`` is set by `k_face_choice`: the value of the donor cell (the cell with the
higher ``\phi``, the one the water flows out of) for `"implicit_upwind"` and `"upwind"`; the arithmetic or
harmonic mean of the two cells for `"arithmetic"` and `"harmonic"`. Taking the donor's depth is
what keeps an empty cell from draining water it does not have. With ``D_f = K_{0,f} |(\nabla\phi)_f| / \nu``,
the Reynolds number is the root of the quadratic of the flux law, and the face flux follows from it:

```math
\mathrm{Re}_f = \frac{2 D_f}{1 + \sqrt{1 + 4 \omega D_f}}, \qquad
K_f = \frac{K_{0,f}}{1 + \omega \mathrm{Re}_f}, \qquad
q_f = -K_f (\nabla\phi)_f .
```

This form avoids cancellation as ``\omega \to 0`` and gives ``|q_f| = \nu \mathrm{Re}_f`` exactly, with no lag
between ``q`` and ``\mathrm{Re}``. The same ``K_f`` is assembled into the matrix and used for every
diagnostic (``\mathbf{q}``, ``\mathrm{Re}``, ``\dot m``, ``\mathcal{D}``), so the water that the solve
conserves and the water that melts the ice are the same. A face to an `OCEAN` or `LAND` cell takes
the grounded cell's ``W``, and a face to `OTHER_BASIN`, `FROZEN_BED` or the edge of the grid has ``K_f = 0``.

## The discrete head equation

For cell ``P`` with neighbours ``F`` across its four faces ``f``, the divergence of the face fluxes is

```math
(\nabla\cdot\mathbf{q})_P = \frac{q_{i+\frac12} - q_{i-\frac12}}{\Delta x} + \frac{q_{j+\frac12} - q_{j-\frac12}}{\Delta y} ,
```

and integrating the mass balance over the cell over the step gives the row of cell ``P``:

```math
\sum_f \frac{K_f}{\Delta_f^2} \left( h_P - h_F \right)
  + \frac{E(h_P) - E(h_P^n)}{\Delta t}
  = \frac{\dot m_P}{\rho_w} + i_{eb,P}
  - \frac{\mathcal{W}_P(h_P) - W^n_P}{\Delta t}
  - \sum_f \frac{K_f}{\Delta_f^2} \left( b^n_P - b^n_F \right) ,
```

where ``\Delta_f`` is ``\Delta x`` or ``\Delta y`` and ``\mathcal{W}_P(h)`` is the water that the cell
holds at the end of the step at head ``h``. By the closure of [Three regimes of the sheet](@ref)

```math
\mathcal{W}_P(h) =
\begin{cases}
\mathcal{G}_P(p_o) - (z_b - h) & h < z_b \quad \text{(unfilled)} \\
\mathcal{G}_P\!\left(p_o - \rho_w g (h - z_b)\right) & z_b \le h \le h_o \quad \text{(filled)} \\
\mathcal{G}_P(0) + (h - h_o) & h > h_o \quad \text{(uplifted)}
\end{cases}
```

with ``\mathcal{G}_P(N)`` the gap after the step at effective pressure ``N``. The three branches agree
at ``h = z_b`` and ``h = h_o``, so a cell fills, pressurizes and lifts the ice continuously. In the
unfilled and uplifted branches ``\mathcal{W}_P`` grows with ``h`` with slope 1, which puts
``1/\Delta t`` on the diagonal; in the filled branch the slope is
``\rho_w g\, |\partial \mathcal{G}_P / \partial N|`` (closure grows with ``N``), computed by a central
difference. The englacial storage ``E(h) = e_v (h - z_b)`` adds ``e_v / \Delta t`` to the diagonal.

Under `ExplicitGapCoupling`, in the filled branch, ``(\mathcal{W}_P - W^n_P)/\Delta t`` is replaced by
``(b^n_P - W^n_P)/\Delta t`` (zero in a full gap) plus the gap rate at ``b^n``,

```math
\frac{\dot m_P}{\rho_i} + \beta(b^n_P)\, |\mathbf{u}_b|_P - A_P |N_P|^{n-1} N_P\, l_c(b^n_P) ,
```

which is the right-hand side of [Head equation](@ref) on the grid, the scheme of [Sommers2018](@cite).

A cell whose four faces are all closed has no neighbour to exchange water with. Its row is solved
exactly: the water at the end of the step, ``W^n + \Delta t (\dot m / \rho_w + i_{eb})``, is known,
and ``\mathcal{W}_P(h) =`` that water is solved for ``h`` by bisection. An `OCEAN` or `LAND`
neighbour's head is known, so it is moved to the right-hand side, which keeps the matrix symmetric
where the face scheme allows it. `OCEAN`, `LAND`, `OTHER_BASIN` and `FROZEN_BED` cells have the row
``h_P = `` their boundary head (or, for the last two, their current head).

## Picard iteration

The row is nonlinear in ``h`` through ``N`` in ``\mathcal{G}``, through ``W`` and ``\mathrm{Re}`` in
``K_f``, and through ``\mathbf{q}`` and ``\boldsymbol{\tau}_b(N)`` in ``\dot m``. Iterate ``k \to k+1``
(`PicardSolver`):

1. Evaluate ``K_f``, ``\dot m`` and the regime of every cell at ``h^k``, and linearize ``\mathcal{W}_P``
   about ``h^k_P``: ``\mathcal{W}_P(h) \approx \mathcal{W}_P(h^k) + \mathcal{W}_P'(h^k)\,(h - h^k)``.
   Under `ExplicitGapCoupling` the creep term is linearized the same way, which puts
   ``n \rho_w g A |N|^{n-1} l_c`` on the diagonal.
2. Under `"implicit_upwind"`, ``K_f`` takes the donor's water depth at the end of the step,
   ``\mathcal{W}_d(h^k)``, and the row also takes the Newton term of the face flux in the donor's head,
   ``\partial |q_f| / \partial h_d = 3 K_f |(\nabla\phi)_f|\, (1 + \omega\mathrm{Re}_f) / \left(W_d (1 + 2\omega\mathrm{Re}_f)\right) \cdot \mathcal{W}_d'``.
   The iteration is then Newton's method in the water depth, which converges at any ``\Delta t``, where
   a lagged depth (`"upwind"`) would be stable only for ``\Delta t < \Delta x\, W / (3 |q|)``. The
   matrix keeps the sign pattern of an M-matrix but is not symmetric.
3. Solve the sparse five-point system for ``h^{k+1}`` with the linear solver ([the tutorial](@ref "8. Solvers")).
4. Limit the update where it crosses a kink of the row (the bed, the overburden head, the head of an
   outflow-only face) so that each solve sees the regime it ends in, and halve an update that reverses
   the previous one. Optionally relax, ``h^{k+1} \leftarrow \alpha h^{k+1} + (1 - \alpha) h^k``.
5. Stop when ``\max |h^{k+1} - h^k| / \max |h^{k+1}| < \mathrm{tol}`` (or ``< \mathrm{atol}``). Where the
   rows store ``\mathcal{W}_P`` exactly (the defaults) it also has to hold that every cell's water
   ``\mathcal{W}_P(h^{k+1})`` matches what its fluxes and sources leave: a limited update can stall the head
   without solving the row.

The first iterate is the head extrapolated in time from the last converged steps
(`head_extrapolation_order`).

## Gap update

`FullyImplicitGapScheme` evaluates the opening and the closure at ``b^{n+1}``:

```math
b^{n+1} = b^n + \Delta t \left( \frac{\dot m}{\rho_i} + \gamma \left(b_r^* - b^{n+1}\right)_+ - C\, b^{n+1} \right),
\qquad \gamma = \frac{|\mathbf{u}_b|}{l_r}, \quad C = A |N|^{n-1} N ,
```

at ``N = N^{n+1}`` and ``\dot m = \dot m^{n+1}``. It is piecewise linear in ``b^{n+1}``, so it is solved in
closed form:

```math
b^{n+1} =
\begin{cases}
\dfrac{b^n + \Delta t\, (\dot m / \rho_i + \gamma b_r^*)}{1 + \Delta t\, (\gamma + C)} & \text{if this is } < b_r^* , \\[2ex]
\dfrac{b^n + \Delta t\, \dot m / \rho_i}{1 + \Delta t\, C} & \text{otherwise.}
\end{cases}
```

Each branch is the step of a linear equation ``\dot b = S - R b``. When its rate ``R`` is negative
(``N < 0``, where creep opens the gap) the update is the exact solution over the step,
``b^n e^{|R|\Delta t} + S (e^{|R|\Delta t} - 1)/|R|``, instead of backward Euler, which would divide by
``1 - \Delta t |R|``. With the cutoff ``l_c = b^2/b_c`` the
branches below ``b_c`` are quadratics, solved the same way. The result is clamped to
``[b_\min, b_\max]``. The update holds for any ``\Delta t``. `ExplicitGapScheme` is forward Euler,
``b^{n+1} = b^n + \Delta t\, \dot b(N^{n+1}, b^n)``, stable only for ``\Delta t\, |\mathbf{u}_b| / l_r < 1``.
Under `WithDiffusion` the term ``\nabla\cdot(\mathcal{D}\nabla b)`` couples the cells, and the gap is solved
as a second linear system, with ``\mathcal{D}`` on the faces averaged from the cells.

## Water and the budget

Under `UnfilledCavities` the water is advanced with the converged fluxes,

```math
W^{n+1}_P = \max\!\left(0,\; W^n_P + \Delta t \left( \frac{\dot m_P}{\rho_w} + i_{eb,P} - (\nabla\cdot\mathbf{q})_P \right) - \left( E(h^{n+1}_P) - E(h^n_P) \right) \right),
```

so the empty part of the gap is ``b^{n+1} - W^{n+1}`` and any water above the gap is the uplift
layer. Otherwise ``W^{n+1} = \mathcal{W}_P(h^{n+1})``. `water_created(sim)` sums, over the grounded
cells, the water the step leaves beyond its sources and fluxes. Under `ConsistentGapCoupling` it is zero
to the tolerance of the Picard iteration.

## Melt on the grid

Each term of ``\dot m`` in a cell is built from its four faces. The flow terms average the face
products over the two faces in each direction,

```math
Q_{\mathrm{diss},P} = \rho_w g \left| \sum_f \tfrac12\, q_f (\nabla\phi)_f \right|, \qquad
Q_{\mathrm{sens},P} = c_t c_w \rho_w \sum_f \tfrac12\, q_f (\nabla p_w)_f ,
```

except that a face to an `OCEAN` or `LAND` cell is credited in full to the grounded cell, so the heat of
the outflow is not lost. ``Q_b`` averages ``u_b \tau_b`` from the faces in the same way, with weight
``\tfrac12``. With `limit_freeze_on` (default) the melt rate is floored at the freeze-on the gap can supply
in the step, ``\dot m \ge -\rho_i \max\!\left(0, (b^n - b_\min)/\Delta t + \beta |\mathbf{u}_b| - C\, l_c\right)``,
with each term evaluated where the gap scheme evaluates it, and under `UnfilledCavities` at most the water
``W^n`` the cell holds.
