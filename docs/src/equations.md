# Model equations

This page describes the continuum model solved by Shakti.jl. The sheet is Shakti's
[Sommers2018](@cite), extended to cavities that are partly empty and to ice that is lifted off the bed,
as in GlaDS-2 [Wells2026](@cite) and [SchoofHewittWerder2012](@cite). Each extension can be turned
off (see [the tutorial](@ref "7. The sheet")), and the equations then reduce to those of the original Shakti.
The [Discretization](@ref) page shows how the equations are solved on the grid.

## Unknowns and geometry

The model is two-dimensional in the horizontal coordinates ``\mathbf{x} = (x, y)``. Between the bed
``z_b(\mathbf{x})`` and the ice base there is a gap of height ``b(\mathbf{x}, t)``, which holds a depth
``W(\mathbf{x}, t)`` of water volume per unit bed area. The unknowns are

| Symbol | Meaning | Unit | `State` field |
|:-------|:--------|:-----|:--------------|
| ``b`` | gap height (space between the bed and the ice base) | m | `b` |
| ``h`` | reduced head, ``h = z_b + (W - b) + p_w / (\rho_w g)`` | m | `h` |

and the derived fields are the water depth ``W`` (water volume per unit bed area, stored in the
`State` field `W`), given by ``b`` and ``h`` through the closure below, and

```math
H = \max\!\left(0,\; z_s - z_b - b\right), \qquad
p_o = \rho_i g H, \qquad
N = p_o - p_w ,
```

where ``z_s`` is the ice surface, ``H`` the ice thickness above the gap, ``p_o`` the overburden pressure,
``p_w`` the water pressure, ``N`` the effective pressure and ``(x)_+ = \max(x, 0)``. Water above the gap
in the uplifted regime lifts the ice without thinning it, so it does not enter ``H`` and leaves ``p_o``
unchanged. It will be useful to also express the overburden as a reduced head,
``h_o = z_b + p_o / (\rho_w g)``.

## Three regimes of the sheet

The reduced head and the water pressure are related by

```math
p_w = \rho_w g \,\operatorname{clamp}\!\left(h - z_b,\; 0,\; \frac{p_o}{\rho_w g}\right), \qquad
W = b + (h - z_b) - \frac{p_w}{\rho_w g} .
```

This one closure covers three regimes:

| Regime | Reduced head | Pressure | Water |
|:-------|:-----|:---------|:------|
| unfilled (cavity partly empty) | ``h < z_b`` | ``p_w = 0``, ``N = p_o`` | ``W = b - (z_b - h) < b`` |
| filled | ``z_b \le h \le h_o`` | ``p_w = \rho_w g (h - z_b)`` | ``W = b`` |
| uplifted (ice lifted off the bed) | ``h > h_o`` | ``p_w = p_o``, ``N = 0`` | ``W = b + (h - h_o) > b`` |

So the bounds ``0 \le p_w \le p_o`` hold, and once ``p_w`` reaches a bound, changes in ``h`` are
changes in the water depth. `FilledCavities()` removes the lower bound and `NoUplift()` the upper one.
With both, Shakti's sheet is recovered: ``W = b`` and ``p_w = \rho_w g (h - z_b)`` without bounds,
so ``N < 0`` (water above the overburden) and ``p_w < 0`` (suction) are both possible.

## Water flux

The water moves as a thin viscous layer driven by the gradient of the hydraulic head ``\hat{h}``, which is related to the reduced head by

```math
\hat{h} = h + b \quad (\texttt{SheetTopPotential}), \qquad \hat{h} = h \quad (\texttt{BedPotential}).
```

With `SheetTopPotential`, ``\hat{h} = z_b + W + p_w / (\rho_w g)`` is the height the water would reach in a borehole
drilled to the bed (filled and uplifted regimes) or the elevation of the water surface (unfilled regime): in the unfilled
regime ``\hat{h} = z_b + W`` and in the uplifted regime ``\hat{h} = h_o + W``. Multiplied by ``\rho_w g`` it is the hydraulic
potential, the energy density of the water, and the water flows to decrease it. The ``+ W`` term accounts for the thickness
of the water layer, as in [Wells2026](@cite) Eq. 7. `BedPotential` leaves it out, ``\hat{h} = h``, as in [Sommers2018](@cite).
The flux per unit width is

```math
\mathbf{q} = -K \nabla \hat{h}, \qquad
K = \frac{g\, W^3}{12 \nu \left(1 + \omega \,\mathrm{Re}\right)}, \qquad
\mathrm{Re} = \frac{|\mathbf{q}|}{\nu} .
```

At small Reynolds numbers this is laminar Poiseuille flow between parallel plates. As ``\mathrm{Re}`` grows,
the factor ``1 + \omega \mathrm{Re}`` takes the flux to the turbulent limit ``|\mathbf{q}| \propto |\nabla\hat{h}|^{1/2}``.
So the same equations describe distributed and channelized drainage, without a separate channel
network [Sommers2018](@cite). The transmissivity is built from the water depth ``W``, which equals
``b`` in a filled gap, so an empty cavity carries no water. Since ``\mathrm{Re}`` depends on
``\mathbf{q}``, the two relations above make an implicit equation for ``|\mathbf{q}|``. Writing
``D = g W^3 |\nabla\hat{h}| / (12 \nu^2)``, the implicit equation is the quadratic equation
``\omega \mathrm{Re}^2 + \mathrm{Re} - D = 0``, whose positive root is

```math
\mathrm{Re} = \frac{2 D}{1 + \sqrt{1 + 4 \omega D}} .
```

### Free-surface layer (optional)

The law above is flow in a duct, between the bed and the ice, with no slip on both walls. The water of an unfilled
cavity has air above it instead, and its surface carries no shear. The mid-plane of a duct carries no shear either, so a
layer of depth ``W`` with a free surface flows as the lower half of a duct of gap ``2W``. Laminar, that gives

```math
\mathbf{q} = -\frac{g\, W^3}{3 \nu} \nabla \hat{h},
```

four times the duct's flux at the same depth. The Reynolds number of the mirrored duct is ``2|\mathbf{q}|/\nu``, and
``\omega`` multiplies it; ``\omega`` stays a property of the walls (a constant Darcy–Weisbach friction factor
``f = 48\omega``). So in an unfilled cell

```math
K = \frac{g\, W^3}{3 \nu \left(1 + 2\omega \,\mathrm{Re}\right)}, \qquad \mathrm{Re} = \frac{|\mathbf{q}|}{\nu} .
```

In the turbulent limit this is ``\sqrt{2}`` times the duct's flux at the same depth, the ratio a force balance gives:
the bed alone holds the layer against the gradient, ``\tau = \rho_w g W |\nabla\hat{h}|``, where the duct's two walls
share ``\rho_w g b |\nabla\hat{h}|``.

[Sommers2018](@cite) and [Wells2026](@cite) keep the duct law in every regime, and so does Shakti by default. The
free-surface law is switched on with `fs_factor = 4` and `fs_omega = 2`. Each cell then carries a weight ``w`` from 0
(duct) to 1 (free surface), and

```math
K = \frac{g\, W^3 \left(1 + (f_K - 1)\, w\right)}{12 \nu \left(1 + \omega (1 + (f_\omega - 1)\, w)\,\mathrm{Re}\right)},
```

with ``f_K`` = `fs_factor`, ``f_\omega`` = `fs_omega`. ``w = 0`` in filled and uplifted cells. In an unfilled cell ``w``
rises with the empty depth ``e = z_b - h`` as a smoothstep, ``w = t^2 (3 - 2t)`` with ``t = \min(e / (\beta b), 1)``,
``\beta`` = `fs_blend` (default 0.65). With ``\beta = 0`` the conductance jumps by ``f_K`` at ``h = z_b``; see
[The free-surface switch](@ref) for why the ramp is at least 0.645 of the gap wide.

## Water mass balance

The water volume per unit bed area changes by divergence of the flux, melt and input from above:

```math
\frac{\partial W}{\partial t} + \frac{\partial b_e}{\partial t} + \nabla \cdot \mathbf{q}
  = \frac{\dot m}{\rho_w} + i_{e\to b} .
```

``\dot m`` [kg m⁻² s⁻¹] is the melt rate of the ice at the base and ``i_{e\to b}`` [m s⁻¹] the water
that enters the bed from above (surface melt through moulins and crevasses, `ieb`). ``b_e`` is the
water stored in the ice above the bed, the englacial storage of [Sommers2018](@cite):

```math
b_e = e_v \frac{p_w}{\rho_w g} = e_v \left( h - z_b - (W - b) \right),
```

with the englacial void ratio ``e_v`` (`e_v = 0` by default). The second form follows from the definition
of ``h``. The closure makes ``W - b`` a function of ``h`` in each regime, so ``b_e`` is a function of ``h``
and ``p_o``:

| Regime | ``W - b`` | ``b_e`` | ``\partial b_e / \partial t`` |
|:-------|:----------|:--------|:------------------------------|
| unfilled | ``h - z_b`` | ``0`` | ``0`` |
| filled | ``0`` | ``e_v (h - z_b)`` | ``e_v\, \partial h / \partial t`` |
| uplifted | ``h - h_o`` | ``e_v\, p_o / (\rho_w g)`` | ``e_v\, \partial p_o / \partial t \,/\, (\rho_w g)`` |

that is, ``b_e = e_v \operatorname{clamp}(h - z_b,\, 0,\, p_o/(\rho_w g))``. Outside the filled range a
change in head is a change in ``W``, not in ``b_e``, so the same water is not counted twice. In an uplifted
cell ``b_e`` still changes when the overburden does, since ``p_o`` depends on ``b`` and ``z_s`` through ``H``:
water drains from the ice to the bed as ``p_o`` falls. On Shakti's sheet (`FilledCavities()`, `NoUplift()`)
there is no clamp and ``b_e = e_v (h - z_b)``.

## Gap height

The gap opens by melting of the ice roof and by sliding over bed bumps, and it closes by viscous
creep of the ice:

```math
\frac{\partial b}{\partial t}
  = \frac{\dot m}{\rho_i}
  + \beta(b)\, |\mathbf{u}_b|
  - A |N|^{n-1} N \, l_c(b)
  \;\left[ + \nabla \cdot (\mathcal{D} \nabla b) \right] .
```

``\mathbf{u}_b`` is the sliding velocity of the ice at its base, ``A`` the ice softness (`A_visc`) and ``n`` the exponent of Glen's law.
The opening by sliding is that of a cavity in the lee of bumps of height ``b_r`` and spacing ``l_r``,

```math
\beta(b) = \max\!\left(0,\; \frac{b_r^* - b}{l_r}\right), \qquad b_r^* = \max\!\left(0,\; b_r - (W - b)_+\right),
```

where ``b_r^*`` is the height of the bumps still in contact with the ice: ice lifted by a layer
``W - b`` rides over the bumps lower than that layer. Without uplift ``b_r^* = b_r``. The creep length scale is ``l_c = b``.
With the cutoff of [Felden2023](@cite) Eq. 7 (`b_c > 0`) it is ``l_c = b^2 / b_c`` for ``b \le b_c``
and ``N > 0``, which slows the closure of thin gaps. At ``N < 0`` the creep term opens the gap.

The term in brackets is optional (`WithDiffusion`). It is the melting of channel walls by the heat
dissipated in the channel, as in Suhmo [Felden2023](@cite) Eq. 10, with

```math
\mathcal{D} = \frac{\min(W, b)}{\rho_i L} \left( Q_\mathrm{diss} + Q_\mathrm{sens} \right)_+ ,
```

where ``Q_\mathrm{diss}`` and ``Q_\mathrm{sens}`` are the two flow terms of the melt rate below.

## Melt rate

The heat available at the bed melts the ice at the rate

```math
L \dot m
  = G
  + \underbrace{|\mathbf{u}_b \cdot \boldsymbol{\tau}_b|}_{Q_b}
  \underbrace{- \rho_w g\, \mathbf{q} \cdot \nabla \hat{h}}_{Q_\mathrm{diss}}
  + \underbrace{c_t c_w \rho_w\, \mathbf{q} \cdot \nabla p_w}_{Q_\mathrm{sens}}
  - q_T ,
```

with the geothermal flux ``G``, the frictional heat ``Q_b`` of sliding with basal shear stress
``\boldsymbol{\tau}_b``, the heat ``Q_\mathrm{diss} = \rho_w g K |\nabla\hat{h}|^2 \ge 0`` dissipated by the
flow, the sensible heat ``Q_\mathrm{sens}`` needed to keep the water at the pressure-melting point
(which falls by ``c_t`` per Pascal), and the heat ``q_T`` conducted into the ice. We assume that water is always at its pressure-melting point temperature, so that water that flows
towards lower pressure, ``\mathbf{q} \cdot \nabla p_w < 0``, is warmed to its rising melting point, which
takes heat from melting. Each term can be turned off (`mdot_includes_*` in `ModelParameters`).
``\boldsymbol{\tau}_b`` comes from a sliding law in ``N`` and ``\mathbf{u}_b`` (table in
[the tutorial](@ref "Sliding law")) or from an ice-flow model. Freeze-on (``\dot m < 0``) is limited
to the water the gap holds.

## Head equation

Subtracting the gap equation from the mass balance eliminates the time derivative of the gap and
gives an equation for the head. In a filled cell (``W = b``) it reads

```math
-\nabla \cdot \left( K \nabla (h + b) \right) + e_v \frac{\partial h}{\partial t}
  = \dot m \left( \frac{1}{\rho_w} - \frac{1}{\rho_i} \right)
  - \beta |\mathbf{u}_b|
  + A |N|^{n-1} N \, l_c
  \left[ - \nabla \cdot (\mathcal{D} \nabla b) \right]
  + i_{e\to b} ,
```

with ``N = p_o - \rho_w g (h - z_b)``. The left side is the net outflow of water plus its englacial
storage. The right side is the net source of water to the cell once the change in gap volume has been
subtracted. Its terms are:

- melt: the water ``\dot m / \rho_w`` produced minus the gap volume ``\dot m / \rho_i`` opened. This is
  negative, since ``\rho_w > \rho_i``: the gap grows by more than the water it receives,
- sliding: ``-\beta |\mathbf{u}_b|``, since the cavities that sliding opens add gap volume but no water,
- creep: ``+A |N|^{n-1} N\, l_c``, the water pushed out as the ice closes the gap,
- the input ``i_{e\to b}``.

With ``e_v = 0`` the equation is elliptic in ``h`` at each time. It is nonlinear because ``N`` enters the
creep term, and because ``\mathrm{Re}`` and ``\dot m`` enter ``K`` and the right side. With
``\hat{h} = h`` it is the head equation of [Sommers2018](@cite).

In an unfilled or uplifted cell the water depth and the gap height differ, ``W \ne b``, and the equation
changes in two ways.

First, ``N`` is fixed (at ``p_o`` in an unfilled cell, at ``0`` in an uplifted one), so the creep term no
longer depends on ``h``.

Second, a storage term appears on the left. The water depth changes in time as
``\partial W / \partial t = \partial b / \partial t + \partial (W - b) / \partial t``. The gap equation gives
``\partial b / \partial t``, and that part is what the equation above already contains. The second part,
``\partial (W - b) / \partial t``, is zero in a filled cell because ``W - b = 0``. It is not zero here, and the
closure ties it to the head: ``W - b = h - z_b`` in an unfilled cell and ``W - b = h - h_o`` in an uplifted
one. So a change in ``W - b`` is a change in ``h``, and the term is a storage term
``\partial h / \partial t`` on the left, like the englacial one but with coefficient 1 instead of ``e_v``.
The englacial term itself no longer depends on ``h`` there (see [Water mass balance](@ref)): it is zero in an
unfilled cell and ``e_v\, \partial p_o / \partial t \,/\, (\rho_w g)``, a source of water, in an uplifted one.

The head equation and the gap equation depend on each other's unknown, so they cannot be solved one after
the other. The head equation needs the gap height ``b``, which sets the conductivity ``K``. The gap equation needs
the effective pressure ``N``, which is set by the head ``h`` through ``N = p_o - \rho_w g (h - z_b)``.
[Discretization](@ref) shows how a time step solves them together.

## Boundary conditions

Every cell has a mask value. The equations above hold on `GROUNDED` cells. The other values set the
boundary conditions:

| Mask | Condition |
|:-----|:----------|
| `OCEAN` | ``h = z_b + \left(p_\mathrm{atm} - \rho_{sw} g \min(z_b, 0)\right) / (\rho_w g)``: the pressure of the ocean at the bed |
| `LAND` | ``h = z_b + p_\mathrm{atm} / (\rho_w g)``: water leaves at atmospheric pressure |
| `OTHER_BASIN` | no flux across the face, ``\mathbf{q}\cdot\mathbf{n} = 0``, i.e. ``\partial \hat{h} / \partial n = 0``; the cell is not solved, so its head ``h`` stays at its initial value for the whole run, which is the one of `LAND`, ``h = z_b + p_\mathrm{atm} / (\rho_w g)`` |
| `FROZEN_BED` | no flux across the face, ``\mathbf{q}\cdot\mathbf{n} = 0``, i.e. ``\partial \hat{h} / \partial n = 0``; the cell is not solved and holds no water: ``b = 0``, ``W = 0``, ``p_w = 0``, so ``h = z_b`` and ``N = p_o`` |
| edge of the grid | no flux, ``\mathbf{q}\cdot\mathbf{n} = 0``, i.e. ``\partial \hat{h} / \partial n = 0`` |

By default `OCEAN` and `LAND` faces are outflow only (`outflow_only_land`, `outflow_only_ocean`): the
face closes when the boundary head is above the hydraulic head ``\hat{h}`` of the grounded cell, so water
can only flow out of the grounded cell into the ocean or land, never from the boundary into the grounded cell.

## Parameters

The default values of `ModelParameters`:

| Symbol | Keyword | Default | Meaning |
|:-------|:--------|:--------|:--------|
| ``\rho_w``, ``\rho_i``, ``\rho_{sw}`` | `rho_w`, `rho_i`, `rho_sw` | 1000, 910, 1027 kg m⁻³ | densities of water, ice and sea water |
| ``g`` | `g` | 9.81 m s⁻² | gravity |
| ``\nu`` | `nu` | 1.787 × 10⁻⁶ m² s⁻¹ | kinematic viscosity of water |
| ``\omega`` | `omega` | 10⁻³ | laminar–turbulent transition [Sommers2018](@cite) |
| ``f_K``, ``f_\omega``, ``\beta`` | `fs_factor`, `fs_omega`, `fs_blend` | 1, 1, 0.65 (off) | free-surface layer in unfilled cavities: conductance and ``\omega`` factors (4 and 2 for the free-surface law), ramp over ``\beta b`` of empty depth |
| ``n`` | `n` | 3 | Glen's law exponent |
| ``L`` | `L` | 3.34 × 10⁵ J kg⁻¹ | latent heat of fusion |
| ``b_r``, ``l_r`` | `br`, `lr` | 0.1 m, 2 m | bump height and spacing |
| ``b_c`` | `b_c` | 0 (``l_c = b``) | creep cutoff [Felden2023](@cite) |
| ``c_t``, ``c_w`` | `ct`, `cw` | 7.5 × 10⁻⁸ K Pa⁻¹, 4220 J kg⁻¹ K⁻¹ | change of melting point with pressure, heat capacity of water |
| ``e_v`` | `e_v` | 0 | englacial void ratio |
| ``p_\mathrm{atm}`` | `p_atm` | 0 Pa | pressure at `LAND` and of the sea surface |
| ``b_\mathrm{min}``, ``b_\mathrm{max}`` | `b_min`, `b_max` | 10⁻⁶ m, ∞ | bounds on ``b`` |
| ``N_\mathrm{min}``, ``N_\mathrm{max}`` | `N_min`, `N_max` | −∞, ∞ | bounds on ``N`` |

The fields ``z_b``, ``z_s``, ``\mathbf{u}_b``, ``A``, ``G``, ``q_T`` and ``i_{e\to b}`` are inputs, set in the
`State` (step 4 of the tutorial) or by a coupled ice-flow model.
