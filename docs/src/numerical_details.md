# Further numerical details

[Discretization](@ref) gives the discrete equations that a step solves. This page covers what surrounds
them, namely the safeguards that let the Picard iteration converge at long time steps, the handling of cells
that exchange no water, the time-step control, changes of the mask, and the linear solvers. Most of it decides
whether, and how fast, a step converges, rather than the head it converges to. Where it does change the
result, as for held donors, the section says so.

## [Upwind donors](@id UpwindDonors)

Under `"implicit_upwind"` each face takes the transmissivity of its donor, the cell with the higher
``\hat{h}``.

### [Donors near a flow divide](@id FlowDivide)

Near a flow divide, where the water on the two sides flows away from each other, or wherever the flow across
a face is about to reverse, the difference of the hydraulic head across the face is small, and its donor can
switch at every iterate. The transmissivity of the face then jumps by orders of magnitude, between that of a thick and that
of a thin cell, although the flux ``-K_f (\nabla\hat{h})_f`` itself passes continuously through zero. The matrix
changes every time, and the iteration can cycle. The donors are therefore tracked through the step in three
stages.

1. At the start of a step every face is free. A face takes the cell with the higher ``\hat{h}`` as its donor at
   its first iterate with flow, and switches each time the flow reverses.
2. After four reversals within the step the face keeps its donor for the rest of the step.
3. Once the head has converged, a face whose held donor has become the downstream cell is moved to its upstream
   side, and the iteration resumes. A face that reverses again, judged at the next
   converged head, takes the transmissivity of its thinner cell. This polishing runs at most three times per step.

The Newton terms of the donors ([Newton terms of the donors](@ref DonorNewtonTerms)) follow the donor in use, so
the matrix stays consistent with the transmissivity it was built from. A face held on its downstream side gets no
Newton term, since there the term would have the wrong sign and put a positive entry off the diagonal. The converged head therefore differs from the upwind solution only where a held face takes the
transmissivity of its downstream cell instead of its upstream one, and only on faces whose flow direction is still
undecided. Both cells of the face read the same flux, so no water is created or lost.

### [Melt in the transmissivity](@id MeltInTransmissivity)

The donor's water thickness is its water at the end of the step at the current head iterate
([Face flux](@ref FaceFlux)). That water includes the melt opening ``\Delta t\, \dot m / \rho_i``, and the
dissipation melt grows with the flux. Taking the melt at the iterate would close the loop
``W \to K \to q \to \dot m \to W`` within the iteration. A larger flux melts more ice, which gives more water and
a still larger flux within the same step. The transmissivity grows with the cube of the water thickness, so a small
extra water thickness ``\delta W`` raises the flux, and roughly the dissipation melt, by ``3\,\delta W/W`` of their
size. Over the step that extra melt opens ``3 \Delta t\, \dot m / (\rho_i W)`` times ``\delta W`` of new water. Where
this factor exceeds one, near large water inputs at long time steps, each iterate amplifies the last, and the
iteration runs away.

The donor's water thickness therefore takes the melt opening from the start of the step, unless the melt at the
iterate gives less water. It then takes the iterate's, so that a draining cell does not pass on water its row no
longer holds. Under `UnfilledCavities` the floor ``W \ge 0`` of the water budget would otherwise make up the
missing water, which creates water ([Conservation](@ref Conservation)). The rows themselves always use the melt at
the iterate, so the water balance of every cell is unaffected, and only the transmissivity lags the melt, by one
step. Under `ExplicitGapCoupling` the donor's water thickness is built from the start-of-step gap, which this
step's melt has not yet opened, so the lag matters only under `ConsistentGapCoupling`.

On the dry floor ([Gap update](@ref GapUpdate)) the water thickness is the water the gap update leaves, not the gap
held at ``b_{\min}``, so a drained cell carries no water. Under `WithDiffusion` it is the smaller of the
start-of-step gap and the gap of the last gap solve. Where the gap opens, the transmissivity therefore takes the
start-of-step gap, so all of the opening lags, not only the melt. Where the gap closes, it takes the gap of the last
gap solve.

## [Limiting the head iterate](@id LimitingTheHead)

Each head row has kinks where its slope jumps, at the bed (unfilled to filled), at the overburden head (filled to
uplifted) and at the boundary head of an outflow-only face ([Boundary cells and faces](@ref BoundaryCellsAndFaces)).

Newton's method can cycle at such a kink. Take one cell with one neighbour of fixed head ``h_F``, a source ``s``,
and ``a = K_f/\Delta x^2``, the weight of the face in the cell's row. The row is
``r(h) = (\mathcal{W}(h) - W^{(m)})/\Delta t + a\,(h - h_F) - s``. Its slope is ``1/\Delta t + a`` on the unfilled
side of ``z_b`` and ``(\mathrm d\mathcal{W}/\mathrm dh)/\Delta t + a`` on the filled side, where
``\mathrm d\mathcal{W}/\mathrm dh`` can be ``10^{-4}`` or less on thick ice
([Derivative of the gap update](@ref GapDerivative)). Where ``a\,\Delta t`` is small, the slope on the filled side is
therefore many times smaller than on the unfilled side. Each row is built
from the regime its head is in at the iterate ``h^{(k)}`` (step 1 of [Picard iteration](@ref PicardIteration)). For
a filled cell the solve therefore uses the filled formula, with its small slope, even for a new head below the bed,
where the unfilled formula, with its large slope, would hold. For this one cell the linear system of that section
reduces to ``h^{(k+1)} - h^{(k)} = -r(h^{(k)}) / (\mathrm dr/\mathrm dh)``, the residual of the row divided by its
slope. From the filled side the small slope therefore gives a large step, which lands far inside the unfilled
regime. From there the large slope of the unfilled formula sends the next step back across the kink, and the
iteration cycles. The same happens at ``h_o`` and at the head where an
outflow-only face opens.

The remedy is to stop an iterate that crosses a kink just inside the new regime. The next solve then uses the
row of that regime, which has no kink, so Newton's method converges from there without jumping back across.

After each linear solve the change of the head goes through three stages.

1. An optional relaxation, off by default. `UnderHeadRelaxation(alpha)` takes only a fraction
   ``0 < \alpha < 1`` of each change, ``h^{(k+1)} \leftarrow \alpha h^{(k+1)} + (1-\alpha) h^{(k)}``.
   `AndersonHeadRelaxation` instead combines the last few changes to extrapolate the next iterate.
2. A stall damping. If the largest change has not fallen below its smallest value so far for 20 iterates, the
   iteration is cycling, and all changes are scaled by a factor ``\omega_s``. The factor is halved each time this
   happens, down to ``\tfrac18``. Both of these stages come before the limits, so that a step stopped at a regime
   boundary lands exactly on it.
3. Limits, cell by cell and in the order below. Here ``\epsilon = \max(10^{-6} b,\, 64\,\mathrm{eps}(|z_b| + 1))``
   is a small step inside a regime, with ``b`` the start-of-step gap and ``\mathrm{eps}(x)`` the floating-point
   spacing at ``x``.

The limits are these.

- Under `ConsistentGapCoupling` the step of a filled grounded cell is capped at
  ``|\Delta N| \le \max(|N|/2,\, p_o/4)``, and at least 1 m of head. On thick ice the gap update closes a thin
  gap within one step for most ``N``, so near ``N = p_o`` the gap it returns barely changes with ``N``, and the
  Newton step overshoots by kilometres. On the filled sheet without uplift, where ``N`` may go negative, ``N`` may
  also fall at most ``p_o/4`` below ``\min(N^{(k)}, 0)`` per iterate. This keeps ``N`` from running away and the
  exponential step of [Negative effective pressure](@ref NegativeN) from overflowing. The cap is off on the filled
  sheet with englacial storage (``e_v > 0``), whose rows stay steep in ``N``.
- An iterate that crosses a regime boundary stops there. A cell is unfilled only while its head is strictly below
  the bed, and at ``h = z_b`` it is filled. A step from filled to unfilled therefore stops at ``z_b - \epsilon``,
  just inside the unfilled regime, and a step from unfilled to filled stops at ``z_b`` itself. In the same way a
  cell is uplifted only while its head is strictly above ``h_o``, so a step from filled to uplifted stops at
  ``h_o + \epsilon``, and a step from uplifted to filled at ``h_o``. Since ``\epsilon`` is tiny, the water
  thickness at the stop is still almost the gap. The regime of every cell, with its ``p_w``, ``N``, water
  thickness, transmissivities and Newton terms, is recomputed from the head at every iterate, so the next solve uses
  the row of the regime the head is now in.
- Inside a regime the step is limited too. Starting from a thin layer, whose transmissivity and Newton terms are
  small, an unlimited Newton step can land far away. In an unfilled cell the depth of the head below the bed
  therefore at most doubles per iterate, or grows by three quarters of the gap if that is more. In an uplifted cell
  the water thickness at most doubles.
- An outflow-only face is open while the head is at or above its boundary head. An update that closes the face
  stops just below the boundary head, at a distance ``\epsilon``, and an update that opens it stops at the boundary
  head. The face is then in the state it ends in for the next solve.
- An update that reverses the previous one by more than half of it is cut to half. Near a root Newton steps shrink
  faster than that, so the limit does not bind there. At a kink it brackets the root and halves the bracket at
  every iterate, as bisection does.

The limits act on the extrapolated first iterate of a step too, except the cap on filled cells
([The first iterate](@ref FirstIterate)).

A converged head is a fixed point of the limited iteration, so the limits do not change the solution. A limit can,
however, keep the step of an iterate tiny while the solve asks for a large one, at a regime or outflow stop or under
the halving of reversed updates. The head test then passes although the cell does not balance its water. The water
test of [Picard iteration](@ref PicardIteration) catches this, because it measures the water balance of the cell
itself rather than the change of the head.

## [The free-surface switch](@id FreeSurfaceSwitch)

The [free-surface layer](@ref "Free-surface layer (optional)") changes the flux law where the head of a cell falls
below the bed and the water gets a free surface. It acts only under `UnfilledCavities`, since only an unfilled
cell has air above its water. On the other sheets its factors have no effect, and building the `Simulation` warns that they need
`UnfilledCavities`. The water of an
unfilled cell has no shear at its top, so it flows like the lower half of a duct twice as thick, whose mid-plane
has no shear either. In laminar flow that layer carries four times the water of a duct of the same water
thickness, and the Reynolds number of the twice-as-thick duct is twice that of the layer.

A cell of weight ``w`` multiplies its laminar transmissivity by ``1 + (f_K - 1)\, w`` and its ``\omega`` by
``1 + (f_{\omega} - 1)\, w`` ([Face flux](@ref FaceFlux)). With `fs_factor` ``= f_K = 4`` and `fs_omega`
``= f_{\omega} = 2``, a full free-surface layer carries four times as much water as a duct of the same water
thickness in laminar flow, and ``\sqrt{2}`` times as much in fully turbulent flow. The layer is off by default
(`fs_factor = fs_omega = 1`).

With a sharp switch (`fs_blend = 0`) the weight jumps from 0 to 1 where the head reaches the bed, so the
transmissivity of a cell jumps by ``f_K``. Near ``h = z_b`` the outflow of a cell then falls as its head rises,
and an unfilled cell drains four times faster than a just-filled one. Take one cell at steady state, fed by an
inflow ``Q`` and draining through one face at a fixed gradient, and suppose ``Q`` lies between what the duct law
and the full free-surface law drain at ``h = z_b``. Then two states are consistent. Filled, the cell drains too
slowly to empty. Unfilled, it drains too fast to fill. The Picard iteration still converges, but the equations of
the step can have two solutions, one each side of the switch, and the step size picks which.

A ramp removes the second state. The empty depth of a cell is ``z_b - h``, the height of the unfilled part of
the gap. `fs_blend` ``= \beta_{\mathrm{fs}}`` ramps the weight from 0 to 1 over an empty depth of
``\beta_{\mathrm{fs}}`` times the gap. With ``\xi = (z_b - h)/(\beta_{\mathrm{fs}} b)`` in ``(0, 1)`` the position
inside the ramp, the water thickness there is ``b(1 - \beta_{\mathrm{fs}} \xi)`` and the weight is
``w(\xi) = \xi^2(3 - 2\xi)``. As the head rises, ``\xi`` falls. The weight drops, which lowers the transmissivity,
while the water thickness grows, which raises it.

The flux of a face is ``|q| = K\,|\nabla\hat{h}|`` with ``K = K_0/(1 + \omega_f \mathrm{Re})``
([Face flux](@ref FaceFlux)), where

```math
K_0 = \frac{g}{12\nu}\,\big(1 + (f_K - 1)\,w\big)\,W^3, \qquad \omega_f = \omega\,\big(1 + (f_{\omega} - 1)\,w\big),
```

and ``\mathrm{Re}`` is the positive root of ``\omega_f \mathrm{Re}^2 + \mathrm{Re} = K_0 |\nabla\hat{h}|/\nu``.
Both ``K_0`` and ``\omega_f`` change with ``\xi``. At a fixed head gradient the transmissivity does not fall as the
head rises if ``\mathrm dK/\mathrm d\xi \le 0`` throughout the ramp. Since ``\mathrm{Re}`` depends on the gradient,
this is checked in the two limits of the flow. Differentiating the equation for ``\mathrm{Re}`` at a fixed gradient
gives

```math
\frac{1}{K}\frac{\mathrm dK}{\mathrm d\xi}
  = (1 - \lambda)\, \frac{1}{K_0}\frac{\mathrm dK_0}{\mathrm d\xi}
  + \lambda\, \frac{1}{2}\left( \frac{1}{K_0}\frac{\mathrm dK_0}{\mathrm d\xi} - \frac{1}{\omega_f}\frac{\mathrm d\omega_f}{\mathrm d\xi} \right),
\qquad \lambda = \frac{2\omega_f \mathrm{Re}}{1 + 2\omega_f \mathrm{Re}} .
```

The first bracket is the relative change of ``K`` in laminar flow, where ``K = K_0``. The second is its relative
change in fully turbulent flow, where ``\mathrm{Re} \approx \sqrt{K_0 |\nabla\hat{h}| / (\nu\, \omega_f)}`` and so
``K \propto \sqrt{K_0/\omega_f}``. The weight ``\lambda`` goes from 0 in laminar flow to 1 in fully turbulent flow,
so the relative change of ``K`` is a weighted mean of the two limits. If neither limit falls as the head rises,
``K`` does not fall at any ``\mathrm{Re}``.

In the laminar limit the condition is ``\mathrm dK_0/\mathrm d\xi \le 0``. With
``\mathrm dw/\mathrm d\xi = 6\xi(1-\xi)`` and ``\mathrm dW/\mathrm d\xi = -\beta_{\mathrm{fs}} b``,

```math
\frac{\mathrm dK_0}{\mathrm d\xi} = \frac{g}{12\nu}\, W^2 b \left[ (f_K - 1)\, 6\xi(1-\xi)\,(1 - \beta_{\mathrm{fs}}\xi) - 3\beta_{\mathrm{fs}} \big(1 + (f_K - 1)\,w\big) \right],
```

and since the factor in front is positive, the condition reads

```math
3\,\big(1 + (f_K - 1)\,w(\xi)\big) \;\ge\; (f_K - 1)\,(1 - \beta_{\mathrm{fs}} \xi)\,\frac{6\xi(1-\xi)}{\beta_{\mathrm{fs}}} \qquad \text{for all } \xi \in (0, 1).
```

This holds for ``\beta_{\mathrm{fs}} \ge 0.645`` at ``f_K = 4``, found numerically, with the bound binding at
``\xi \approx 0.23``. In the fully turbulent limit the condition is
``\mathrm d(K_0/\omega_f)/\mathrm d\xi \le 0``, which is weaker (``\beta_{\mathrm{fs}} \ge 0.406`` with
``f_{\omega} = 2``), since ``\omega_f`` grows with the weight too and offsets part of the rise of ``K_0``.
`fs_blend_monotone` checks both limits and returns the stricter bound, for any factors. The default `fs_blend = 0.65` lies just above it. A narrower ramp gives a
warning, once per run, that a cell can then have two steady states and that ``\Delta t`` can decide which one a
run reaches. With the ramp the transmissivity never falls as the head rises, and the mechanism above cannot give two
states. Below the bound, where the transmissivity does fall as the head rises, the Newton term of the donor
([Newton terms of the donors](@ref DonorNewtonTerms)) would be negative, and it is kept at 0.

## [The first iterate](@id FirstIterate)

The iteration of a step starts from a first iterate ``h^{(0)}``. Starting from the last solution,
``h^{(0)} = h^{(m)}``, would assume that the head does not change over the step. By default Shakti.jl instead
extrapolates linearly in time through the heads converged at the end of the last two steps, to the end of the
coming step,

```math
h^{(0)} = h^{(m)} + \frac{\Delta t}{t^{(m)} - t^{(m-1)}} \left( h^{(m)} - h^{(m-1)} \right).
```

The factor ``\Delta t/(t^{(m)} - t^{(m-1)})`` is 1 at a constant time step and accounts for a change of step under
the adaptive time step. The extrapolation changes only where the iteration starts, not the solution. Where the
head varies smoothly in time, it starts closer and needs fewer iterates.

`head_extrapolation_order` sets the degree ``n_e`` (1 by default, and 0 for no extrapolation,
``h^{(0)} = h^{(m)}``). The extrapolation then goes through the last ``n_e + 1`` converged heads. Its weights are
those of the polynomial of degree ``n_e`` through these heads at their actual times ``t^{(m)}, t^{(m-1)}, \dots``,
so unequal steps need no special treatment. Only converged heads enter the history. The initial condition does not
count, since it need not solve the rows, so the first linear extrapolation happens at the third step. A step that
does not converge clears the history. Until enough heads exist, the highest degree they allow is used, down to
``h^{(0)} = h^{(m)}``. A checkpoint saves the history, so a restarted run extrapolates from its first step. Only
grounded cells are extrapolated. Land and ocean cells have fixed heads, and cells in another basin or on a frozen
bed keep their current head, which an extrapolation would make drift.

The extrapolated head is then limited like every iterate ([Limiting the head iterate](@ref LimitingTheHead)),
measured from the last converged head. A cell that changed regime in the last step moved by tens of metres.
Extrapolating that move into the uplifted regime would start the step with an uplift layer, water lifting the ice,
tens of metres thick, whose dissipation melt then feeds the layer faster than the iteration can drain it. The cap
on filled cells is left out here, since on the first iterate it doubled the number of iterates in tests on the dry
floor, and the first iterate is only a starting point. The halving of reversed updates does not act on the first
iterate, since the record of the previous update is reset at the start of each step. It would otherwise compare
with the last update of the previous step.

## [Closed cells](@id ClosedCells)

A grounded cell whose four faces are all closed exchanges no water with its neighbours. Its faces can be
closed because its neighbours are frozen or in another basin, because it lies on the edge of the grid, or
because an outflow-only face to land or ocean carries no flow. Without englacial storage (``e_v = 0``), the
total water thickness of a cell is its water thickness alone, ``\mathcal{W}_P(h)``, and the row of a closed cell keeps
only the term ``(\mathrm d\mathcal{W}_P/\mathrm dh)/\Delta t``, the change of that water with its head. Near
``N = 0`` the creep closure grows like ``N^n``, so the water thickness hardly changes with the head, and at
``N = 0`` it does not change at all. The linearized row then cannot fix the head. A small error in the water
becomes a large jump in the head, or the row is zero. The head of a closed cell is therefore found
directly, from the water the cell must hold at the end of the step.

Since no water crosses its faces, that water is known before the solve,

```math
T = W^{(m)} + \Delta t \left( \frac{\dot m}{\rho_w} + i_{e\to b} \right),
```

and under `WithDiffusion` the water that wall diffusion moves out of the cell is subtracted from it. The head
of the cell is the one at which the cell holds ``T``. Under `ConsistentGapCoupling` the water at the bed is
set by the gap update ``\mathcal{G}(N)``, which decreases as ``N`` grows, since a higher effective pressure closes
the gap faster. A full gap therefore holds at least ``\mathcal{G}(p_o)``, the gap closed by the full
overburden, and at most ``\mathcal{G}(0)``, the gap with no closure. Where ``T`` falls relative to these two
values decides the case.

- If ``T`` is more than even the open gap ``\mathcal{G}(0)`` can hold and uplift is allowed, the ice is
  lifted by the excess, ``h = h_o + T - \mathcal{G}(0)``.
- If ``T`` is less than even the most closed gap ``\mathcal{G}(p_o)`` holds and unfilled cavities are allowed,
  the gap is partly empty by the shortfall, ``h = z_b - (\mathcal{G}(p_o) - T)``.
- Otherwise the gap is full, and ``\mathcal{G}(N) = T`` is solved for ``N`` by bisection, which always
  converges because ``\mathcal{G}`` is monotone. Then ``h = z_b + (p_o - N)/(\rho_w g)``, with ``N`` kept
  within ``[N_{\min}, N_{\max}]``.

Under `ExplicitGapCoupling` the gap does not change within the step, and the row of a closed cell is a steady
balance. Creep must close the gap as fast as sliding and melt open it,

```math
A\, |N|^{n-1} N\, l_c = R, \qquad R = \beta\, |\mathbf{u}_b| - \dot m \left( \frac{1}{\rho_w} - \frac{1}{\rho_i} \right) - i_{e\to b},
```

which has the exact solution ``N = \operatorname{sign}(R)\, \big( |R| / (A\, l_c) \big)^{1/n}``, kept within
``[N_{\min}, N_{\max}]``. With unfilled cavities or uplift, ``R`` also includes the empty part of the gap at
the start of the step, which the water must fill first, or the uplift water, which is released first. The
same two end cases are taken before this one. The gap is partly empty if even the closure at ``N = p_o`` is
faster than the opening, and the ice is lifted if ``R < 0``, since then the gap cannot take the water even
without closure. Without creep (``A\, l_c = 0``) the gap cannot close, so ``N = p_o`` if ``R > 0`` and
``N = 0`` otherwise.

With ``e_v > 0`` the englacial storage keeps every row's diagonal positive, so closed cells take the ordinary
row. A patch of several cells that is closed as a whole, with all its cells at ``N = 0``, has no head level of
its own, and is not covered.

## [Time-step control](@id TimeStepControl)

Under `FixedTimeStep` every step has the same ``\Delta t``. A step whose head solve does not converge is taken as
it is, counted in `water_created(sim).unconverged`, and reported once with the water it created.

`AdaptiveTimeStep(grid, state; dt_min, dt_max, target_time, ...)` changes ``\Delta t`` from step to step. The
first step takes the `dt` of the `Simulation`, kept within ``[\Delta t_{\min}, \Delta t_{\max}]``. After that,
``\Delta t`` changes in four ways.

- A step whose head solve does not converge, at ``\Delta t > \Delta t_{\min}``, is rejected and retried from its
  start at half the step, no less than ``\Delta t_{\min}``. Only the head needs resetting, since the gap, the
  water, the start-of-step melt and the head history change only once a step is accepted. The freeze-on floor
  is recomputed at the shorter step, and the count of fast steps below starts again. At ``\Delta t_{\min}`` an
  unconverged step is no longer retried and is taken as it is, as under `FixedTimeStep`.
- After `grow_after` (3) accepted steps in a row that each converged within `fast_iters` (30) iterates, the step
  grows by the factor `grow` (1.5), up to ``\Delta t_{\max}``.
- With `safety_factor` set, the step is capped at ``\Delta t \le \text{safety\_factor} / R``, no lower than
  ``\Delta t_{\min}``. Here ``R`` is the relaxation rate of the gap update
  ([Accuracy of the gap update](@ref GapAccuracy)), with ``C`` multiplied by ``\mathrm dl_c/\mathrm db`` under
  `CreepCutoff`. It is taken at the start of the step, from the last converged ``N`` and sliding speed, over the
  grounded cells above ``b_{\min}``. By default ``R`` is its largest value over these cells. With
  `use_percentile = true` it is instead the `percentile` quantile (0.75 by default) of ``R`` itself, so that three
  quarters of the cells relax more slowly than the cap assumes, and a single fast cell does not set the step. A
  value of 0.3 for `safety_factor` was found by testing on ice sheets and a valley glacier to keep the gap
  accurate.
- The last step is cut to end exactly at `target_time`. Where the time left after a step would be less than
  ``\Delta t_{\min}``, that step is shortened so that the last one is exactly ``\Delta t_{\min}``.

The run stops once it reaches `target_time`, but `tsteps` of the `Simulation` still bounds the number of steps.
It should be at least the remaining time divided by ``\Delta t_{\min}``, rounded up, or the run can stop before
`target_time`.

The gap update has no time-step limit for ``C \ge 0`` ([Accuracy of the gap update](@ref GapAccuracy)), and with
`"implicit_upwind"` faces the linearized head equation has none either
([Why the water thickness is differentiated](@ref WhyDifferentiate)). No bound is proven for the full nonlinear
step with the coefficients it holds at the iterate. In practice the time step is limited by the convergence of the
head solve, which the rejection above handles.

## [Freeze-on capacity](@id FreezeOnCapacity)

With `limit_freeze_on` (the default) the melt rate is floored, ``\dot m \ge \dot m_{\min}``
([Melt on the grid](@ref MeltOnTheGrid)), so that a freezing cell cannot freeze its gap shut below ``b_{\min}``,
or freeze more water than it holds, within one step. Without the floor the gap update would reach its clamp at
``b_{\min}``, and the water the rows freeze would no longer match the gap. The floor is computed once per step,
from the start-of-step gap, and again when a rejected step is retried at a shorter ``\Delta t``
([Time-step control](@ref TimeStepControl)). It caps ``\dot m`` both as the water source of the rows and in the
opening of the gap. The heat terms ``Q_b``, ``Q_{\mathrm{diss}}`` and ``Q_{\mathrm{sens}}`` are stored without it.

The freeze-on capacity ``f_{\max}`` is the largest freezing rate, in metres of ice per second, that the gap
update can absorb within one step without the gap falling below ``b_{\min}``. Setting ``b^{(m+1)} = b_{\min}`` in
the backward-Euler step of [Gap update](@ref GapUpdate) and solving for the melt term ``-\dot m/\rho_i`` gives

```math
f_{\max} = \max\!\left(0,\; \frac{b^{(m)} - b_{\min}}{\Delta t} + \beta(b_{\min})\, |\mathbf{u}_b| - C\, l_c(b_{\min})\right),
```

with ``C = A |N|^{n-1} N`` and ``\beta(b) = \max\!\big(0, (\mathcal{B} - b)/l_r\big)`` the opening by sliding of
[Gap update](@ref GapUpdate). This is the room in the gap above its floor, spread over the step, plus the room that
sliding opens, minus the room that creep closes anyway. The gap equation opens the gap by ``\dot m/\rho_i``, a rate
of ice, so the floor on the melt rate takes the density of ice, ``\dot m_{\min} = -\rho_i f_{\max}``. The formula is
exact for the two backward-Euler cases of the gap update. Where ``N < 0`` the update is the exponential step, and
the formula is a first-order estimate.

With unfilled cavities the capacity is also capped by the water present,
``f_{\max} \le (\rho_w/\rho_i)\, W^{(m)}/\Delta t``, since freezing consumes water that an empty cavity
does not hold. A freezing rate ``-\dot m`` turns ``-\dot m/\rho_w`` of water per second into ``-\dot m/\rho_i`` of
ice, so freezing all of ``W^{(m)}`` within the step means ``-\dot m/\rho_i = (\rho_w/\rho_i)\, W^{(m)}/\Delta t``.
With a water-filled gap, ``W = b``, the room in the gap is already the tighter limit.

Each term is evaluated where the gap scheme evaluates it. The formula above is that of
`FullyImplicitGapScheme`, which evaluates the opening by sliding and the creep at the end-of-step gap, here
``b_{\min}``. `ExplicitGapScheme` evaluates them at ``b^{(m)}``. Under `WithDiffusion` the capacity takes the
local terms of the gap system and ignores the diffusion, which only moves gap volume between cells. The capacity is
then conservative where the gap is a local minimum, which diffusion fills, and generous where it is a local maximum,
which diffusion drains. On the dry floor ([Gap update](@ref GapUpdate)) the update moves the water instead of the gap
and may drain it to zero. The capacity is then

```math
f_{\max} = \max\!\left(0,\; \min\!\left( \frac{W^{(m)}}{\Delta t} + \beta\, |\mathbf{u}_b| - C\, l_c(W^{(m)}),\;
  \frac{\rho_w}{\rho_i} \frac{W^{(m)}}{\Delta t} \right)\right),
```

with the creep term only under `ExplicitGapScheme`, since the fully implicit update evaluates creep at the
end-of-step water, here zero. A cell then freezes no more than its water leaves room for, nor more water than it
holds. On the dry floor the cap by the water always applies.

A host ice model can ask for the capacity over its own step (`freeze_on_capacity!(C, sim; dt)`), when Shakti.jl
takes several steps within one step of the host. The room in the gap is a stock, available once per host step, so
it is spread over the host's step. Spreading it over Shakti.jl's shorter step would overstate the capacity by the
ratio of the two steps. Creep and sliding are then evaluated at the current gap, since over many Shakti.jl steps
they act on the actual gap, not on ``b_{\min}``. Outside the dry floor, `water_limited = false` drops the cap by the
water. With [Melt in the gap opening](@ref GapOpeningMelt) the floor applies to the melt that opens the gap.

## [Melt in the gap opening](@id GapOpeningMelt)

By default all of the melt opens the gap, ``\dot m/\rho_i``, as in [Sommers2018](@cite). In a linked-cavity bed,
however, the ice rests on bedrock contacts, the tops of the bumps. Heat spread evenly over the bed melts the ice
at the contacts as well as at the cavity roofs, and the ice sinks onto the contacts as fast as the roofs rise.
That melt makes water but no gap. `ModelParameters(gap_opening_melt = :exclude_uniform)` (step 5 of
[the tutorial](@ref "5. Parameters")) opens the gap with

```math
\dot m_{\mathrm{open}} = \dot m - w_c\, \frac{\max(0,\, Q_u)}{L}, \qquad Q_u = Q_G + Q_b - Q_T,
\qquad w_c = \operatorname{clamp}\!\left(\frac{\mathcal{B} - b}{\mathcal{B}},\, 0,\, 1\right) \ (0 \text{ if } \mathcal{B} = 0),
```

in place of ``\dot m`` in every gap update. These are the closed form of [Gap update](@ref GapUpdate), the head
rows, the melt in the transmissivity and the gap-diffusion system. The water source ``\dot m/\rho_w``, the water
budget and the ``\dot m`` a host reads keep the full melt, and the diffusion coefficient ``\mathcal{D}`` is
unchanged. Each term of ``Q_u`` enters only if it is part of the melt rate. Four choices define the option.

- The dissipation and the sensible heat are released in the water, at the cavity roofs, and always open the gap.
- Only net uniform melt is left out. Net uniform freezing (``Q_u < 0``, for example at a cold margin) can freeze
  only water, onto the roofs, so it still closes the gap. This is itself a simplification, since the heat
  conducted away through the contacts cools the ice there and freezes nothing.
- The argument needs the ice to rest on its bumps. The weight ``w_c`` is a linear ramp from 1, where the ice sits
  on the bed, to 0, where the gap reaches the bump height ``\mathcal{B}``. Here ``\mathcal{B}`` is the height of the
  bumps under the ice base, `br` minus any uplift layer, so uplifted ice has ``w_c = 0`` and all of its melt opens
  the gap. Without bumps (`br = 0`) the option changes nothing. The ramp smooths the switch between resting and not
  resting on the bumps. It is not a contact fraction derived from the bed geometry. Like ``\mathcal{B}``, ``w_c``
  is taken at the start-of-step gap, so the gap update stays piecewise linear in ``b`` and keeps its closed form.
- Frictional heat is made at the contacts, where on its own it would lower the ice onto the bumps and close the
  cavities at the rate ``Q_b/(\rho_i L)``. Counting it in ``Q_u`` treats it as neutral for the gap. This lies
  between [Sommers2018](@cite), where it opens the gap, and that closure, which is not modelled.

The freeze-on floor ([Freeze-on capacity](@ref FreezeOnCapacity)) applies to ``\dot m_{\mathrm{open}}``, and then
``\dot m = \dot m_{\mathrm{open}} + w_c \max(0, Q_u)/L \ge \dot m_{\mathrm{open}}``. Flooring ``\dot m`` instead
would let ``\dot m_{\mathrm{open}}``, which is ``\dot m`` minus the excluded part, fall below the floor and close
the gap past the capacity. Flooring ``\dot m_{\mathrm{open}}`` keeps the gap above ``b_{\min}``, and
``\dot m \ge \dot m_{\mathrm{open}}`` then keeps the frozen water within the capacity as well. With unfilled
cavities the capacity is already the smaller of the room in the gap and the water the cell holds, so one floor
keeps both the gap above ``b_{\min}`` and the frozen water within the water present.

The effect is largest where the opening by sliding is small next to the uniform melt, about 5 mm/a of ice for
``Q_G = 0.05`` W/m². Without sliding the gap opens only by dissipation, and ``N`` falls toward 0 where the flux
is small.

## [Mask changes](@id MaskChanges)

Between steps, a coupled ice model, which we call the host, can change which cells Shakti.jl solves. Two things change the mask. The
host's own mask decides where there is grounded ice, and the basal temperature decides where that ice is frozen
to its bed. Both changes happen on the time scale of the host, not every hydrology step.

A cell freezes when its basal temperature relative to the pressure-melting point falls below `T_freeze`
(default ``-1`` K), and a frozen cell thaws once it reaches `T_freeze + T_hysteresis`, which is ``-0.5`` K
by default, since `T_hysteresis` defaults to ``0.5`` K. Between the two thresholds a cell keeps its state, so a
temperature that hovers near one threshold does not switch the cell back and forth, resetting its gap each time. `T_hysteresis = 0` gives a single
threshold.

A freezing cell becomes `FROZEN_BED` with ``b = W = 0`` and ``p_w = 0``, so ``h = z_b`` and ``N = p_o``.
Its head is then fixed, and its faces are closed (see [Boundary cells and faces](@ref BoundaryCellsAndFaces)).
The water the cell held, the filled gap, any uplift layer and its englacial water, is discarded. Freezing is
a cutoff, not a model of phase change. The water of a cooling cell is refrozen gradually, before the cell freezes, by the energy
balance of the melt rate, in which the conductive heat flux ``Q_T`` makes ``\dot m`` negative and closes the
gap (see [Freeze-on capacity](@ref FreezeOnCapacity)). What is left when the cell crosses `T_freeze` is
removed without releasing latent heat, since the latent heat and the refrozen ice belong to the host's energy
and mass budgets. A colder `T_freeze` gives the energy balance longer to dry the cell and leaves less to
discard. The discarded water is returned as `discarded_b`, the sum of ``W + b_e`` over the frozen cells
(multiply by ``\Delta x \Delta y`` for a volume). It is not part of the created water of [Conservation](@ref Conservation), which
measures the error of the scheme.

A thawing cell becomes `GROUNDED` again at ``b = b_{\min}`` with no water, an empty cavity or a dry floor, and
with no englacial water. Its head is left at the value it had while frozen and is the starting guess of the next solve. With
``b_{\min} = 0`` a thawed block of cells has all-zero rows in its interior, and the system is singular, so runs
that thaw regions need ``b_{\min} > 0``.

A cell that leaves the grounded set of the host loses its gap and water in the same way, and they are added to
`discarded_b`. It takes the prescribed pressure of its new category. A cell that joins the grounded set starts
like a thawed cell, and its head starts from the pressure of its old category. A frozen cell stays frozen while
the host keeps it grounded.

Freezing can cut a grounded cell off from every grounded, land or ocean neighbour. All its faces are then
closed, its water has nowhere to go, and where its creep term vanishes its row is zero. Such a cell is frozen
too, after the other changes. Freezing it cannot isolate another cell, since it has no grounded neighbour, so
one pass suffices. Its water is added to `discarded_b`, and the number of these cells is returned as
`n_isolated`.

The functions are `update_frozen_mask!` (from the basal temperature), `set_mask!` (from the host's mask), and
the lower-level `freeze_cells!` and `thaw_cells!`. Each refreshes the face masks, ``N``, ``\beta`` and ``l_c``.
Their `Simulation` methods also bring the water within what the sheet allows and refresh the fields that
depend on the options of the simulation, such as the cell clamps on ``N`` and the outflow-only faces.

## [Linear solvers](@id LinearSolvers)

The head system has ``n = n_x n_y`` unknowns and a fixed five-point pattern, each cell coupled to its four
neighbours. With every face rule except `"implicit_upwind"` the matrix is symmetric. Each face has one
transmissivity, which both of its cells read, and the known head of an `OCEAN` or `LAND` neighbour moves to the
right-hand side. On the grounded cells it solves, the matrix is a weighted graph Laplacian plus a diagonal that
is never negative. That diagonal holds the transmissivity of each open `OCEAN` or `LAND` face, the englacial storage
``e_v/\Delta t`` and the term ``(\mathrm d\mathcal{W}/\mathrm dh)/\Delta t`` of the gap or cavity. The matrix is therefore positive semidefinite. It is
positive definite if every group of grounded cells connected through open faces (``K_f > 0``) has one cell where
that diagonal is positive, a cell with an open face to `OCEAN` or `LAND`, with ``e_v > 0``, in the unfilled or
uplifted regime, or with a gap that still responds to ``N``. An enclosed group with none of these (``e_v = 0``,
every cell at ``N = 0`` or with its gap held at a bound) makes it singular, with its head fixed only up to a
constant. Closed cells and `OTHER_BASIN` and `FROZEN_BED` cells have identity rows and do not affect this.

The Newton terms of `"implicit_upwind"` make the matrix non-symmetric. It keeps the sign pattern of an M-matrix,
a positive diagonal and no positive entries off it, and stays non-singular under the same condition, but it is
not positive definite in general. Cholesky factorization and the conjugate-gradient method (CG) therefore apply
only to the symmetric face rules. The solvers are listed below.

| Solver | Matrix it solves | Representation | Fields on |
|:-------|:-----------------|:---------------|:----------|
| `LUDirectSolver` | any | assembled | CPU |
| `CholeskyDirectSolver` | symmetric positive definite | assembled | CPU |
| `CUDSSDirectSolver` | any (`structure = :general`, LU) or symmetric positive definite (`:spd`, Cholesky) | assembled on the CPU, factorized on an NVIDIA GPU | CPU |
| `CGIterativeSolver(grid, SparseAssembledLinearSystem)` | symmetric positive definite | assembled | CPU |
| `CGIterativeSolver(grid, MatrixFreeLinearSystem)` | symmetric positive definite | matrix-free | CPU or GPU |
| `BiCGStabIterativeSolver` | any | matrix-free | CPU or GPU |

An assembled solver builds the sparse matrix explicitly, which only the CPU can do, so its fields stay on the
CPU. Its sparsity pattern is built once, so each iterate only rewrites the values and refactorizes. A sparse
factorization costs about ``n^{3/2}`` operations on a two-dimensional grid, and its memory limits the grid size.
`LUDirectSolver(grid; lagged = true)` reuses its last factorization as the preconditioner of a GMRES solve and
refactorizes only when GMRES stalls. A matrix-free solver never forms the matrix. It applies the five-point
stencil from the per-cell coefficients, so it runs wherever the fields live, and it is the only kind that runs
with the fields on a GPU.

The iterative solvers cost of order ``n`` operations per iteration and use multigrid preconditioners, so their
number of iterations barely grows with the grid. CG on the assembled matrix uses algebraic multigrid, rebuilt
every five solves (`amg_refresh_every`). A matrix-free solver uses an aggregation-multigrid V-cycle by default.
It merges 2 × 2 cells per level and builds each coarse matrix from the fine one (a Galerkin product), so the
coarse levels follow jumps in ``K_f``. Jacobi is the alternative, and CG also takes Chebyshev. The iterative
solvers warm-start from the current head and stop at a relative residual of ``10^{-4}`` (`rtol`), with no
absolute tolerance. Under `WithDiffusion` the gap system is symmetric positive definite, and it takes its own
solver (Cholesky, cuDSS with `:spd`, or CG). See [the tutorial](@ref "8. Solvers") for when to use which.

## Diagnostic fields

Some fields are computed for output only and never read by the solve. They are the cell-centred transmissivity
``K = g b^3 / (12 \nu (1 + \omega \mathrm{Re}))``, the cell Reynolds number, the mean of its four faces, and
the face gaps `b_x`, `b_y`, the mean of the two cells (the nearest cell's gap on the domain edge). The solve
reads the face transmissivities `K_x`, `K_y` built from the water thickness.
