# Model equations

This page states the continuum model that Shakti.jl solves. The sheet is Shakti's
[Sommers2018](@cite), extended to cavities that are partly empty and to ice that is lifted off the bed,
as in GlaDS-2 [Wells2026](@cite) and [SchoofHewittWerder2012](@cite). Each extension can be turned
off (see [the tutorial](@ref "7. The sheet")), and the equations then reduce to those of Shakti.
The [Discretization](@ref) page shows how the equations are solved on the grid.

## Unknowns and geometry

The model is two-dimensional in the horizontal coordinates ``\mathbf{x} = (x, y)``. Between the bed
``z_b(\mathbf{x})`` and the ice base there is a gap of height ``b(\mathbf{x}, t)``, which holds a depth
``W(\mathbf{x}, t)`` of water per unit bed area. The unknowns are

| Symbol | Meaning | Unit | `State` field |
|:-------|:--------|:-----|:--------------|
| ``b`` | gap height (space between the bed and the ice base) | m | `b` |
| ``W`` | water depth (water volume per unit bed area) | m | `W` |
| ``h`` | hydraulic head, referenced to the bed | m | `h` |

and the derived fields are

```math
H = \max\!\left(0,\; z_s - z_b - b - (W - b)_+\right), \qquad
p_o = \rho_i g H, \qquad
N = p_o - p_w ,
```

where ``z_s`` is the ice surface, ``H`` the ice thickness above the gap and the lifted water,
``p_o`` the overburden pressure, ``p_w`` the water pressure, ``N`` the effective pressure and
``(x)_+ = \max(x, 0)``. It is useful to write the overburden as a head,
``h_o = z_b + p_o / (\rho_w g)``.

## Three regimes of the sheet

The head and the water pressure are related by

```math
p_w = \rho_w g \,\operatorname{clamp}\!\left(h - z_b,\; 0,\; \frac{p_o}{\rho_w g}\right), \qquad
W = b + (h - z_b) - \frac{p_w}{\rho_w g} .
```

This one closure covers three regimes:

| Regime | Head | Pressure | Water |
|:-------|:-----|:---------|:------|
| unfilled (cavity partly empty) | ``h < z_b`` | ``p_w = 0``, ``N = p_o`` | ``W = b - (z_b - h) < b`` |
| filled | ``z_b \le h \le h_o`` | ``p_w = \rho_w g (h - z_b)`` | ``W = b`` |
| uplifted (ice lifted off the bed) | ``h > h_o`` | ``p_w = p_o``, ``N = 0`` | ``W = b + (h - h_o) > b`` |

So the bounds ``0 \le p_w \le p_o`` hold, and once ``p_w`` reaches a bound, changes in ``h`` are
changes in the water depth. `FilledCavities()` removes the lower bound and `NoUplift()` the upper one.
With both, Shakti's sheet is recovered: ``W = b`` and ``p_w = \rho_w g (h - z_b)`` without bounds,
so ``N < 0`` (water above the overburden) and ``p_w < 0`` (suction) are both possible.

## Water flux

The water moves as a thin viscous layer driven by the gradient of the potential

```math
\phi = h + b \quad (\texttt{SheetTopPotential}), \qquad \phi = h \quad (\texttt{BedPotential}),
```

the head of the top of the water, in metres. In the unfilled regime ``\phi = z_b + W`` and in the
uplifted regime ``\phi = h_o + W``. ``\rho_w g \phi`` is the potential of [Wells2026](@cite) Eq. 7.
`BedPotential` is the potential of [Sommers2018](@cite), which leaves out the thickness of the water layer.
The flux per unit width is

```math
\mathbf{q} = -K \nabla \phi, \qquad
K = \frac{g\, W^3}{12 \nu \left(1 + \omega \,\mathrm{Re}\right)}, \qquad
\mathrm{Re} = \frac{|\mathbf{q}|}{\nu} .
```

At small Reynolds numbers this is laminar Poiseuille flow between parallel plates. As ``\mathrm{Re}`` grows,
the factor ``1 + \omega \mathrm{Re}`` takes the flux to the turbulent limit ``|\mathbf{q}| \propto |\nabla\phi|^{1/2}``.
So the same equations describe distributed and channelized drainage, without a separate channel
network [Sommers2018](@cite). The transmissivity is built from the water depth ``W``, which equals
``b`` in a filled gap, so an empty cavity carries no water. Since ``\mathrm{Re}`` depends on
``\mathbf{q}``, the two relations above make an implicit equation for ``|\mathbf{q}|``. Writing
``D = g W^3 |\nabla\phi| / (12 \nu^2)``, it is the quadratic
``\omega \mathrm{Re}^2 + \mathrm{Re} - D = 0``, whose positive root is

```math
\mathrm{Re} = \frac{2 D}{1 + \sqrt{1 + 4 \omega D}} .
```

## Water mass balance

The water per unit bed area changes by divergence of the flux, melt and input from above:

```math
\frac{\partial W}{\partial t} + \frac{\partial E}{\partial t} + \nabla \cdot \mathbf{q}
  = \frac{\dot m}{\rho_w} + i_{eb} .
```

``\dot m`` [kg m⁻² s⁻¹] is the melt rate of the ice at the base and ``i_{eb}`` [m s⁻¹] the water
that enters the bed from above (surface melt through moulins and crevasses, `ieb`). ``E`` is the
water stored in the ice above the bed, ``E = e_v (h - z_b)`` with the englacial void ratio ``e_v``
([Sommers2018](@cite) Eq. 13, `e_v = 0` by default). Where the unfilled and uplifted regimes are
allowed, ``E`` takes ``h - z_b`` clamped to ``[0, p_o/(\rho_w g)]``: outside that range the
water is in ``W``.

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

``\mathbf{u}_b`` is the sliding velocity, ``A`` the ice softness (`A_visc`) and ``n`` the exponent of Glen's law.
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
  \underbrace{- \rho_w g\, \mathbf{q} \cdot \nabla \phi}_{Q_\mathrm{diss}}
  + \underbrace{c_t c_w \rho_w\, \mathbf{q} \cdot \nabla p_w}_{Q_\mathrm{sens}}
  - q_T ,
```

with the geothermal flux ``G``, the frictional heat ``Q_b`` of sliding with basal shear stress
``\boldsymbol{\tau}_b``, the heat ``Q_\mathrm{diss} = \rho_w g K |\nabla\phi|^2 \ge 0`` dissipated by the
flow, the sensible heat ``Q_\mathrm{sens}`` needed to keep the water at the pressure-melting point
(which falls by ``c_t`` per pascal), and the heat ``q_T`` conducted into the ice. Water that flows
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
  + i_{eb} ,
```

with ``N = p_o - \rho_w g (h - z_b)``. On the left is the outflow of water and its englacial storage;
on the right is the water that the gap cannot take up: the melt water less the room the melt makes
(negative, since ``\rho_w > \rho_i``), less the room that sliding opens, plus the water that creep pushes out, plus the input.
With ``e_v = 0`` the equation is elliptic at each time and nonlinear in ``h``, through ``N`` in the
creep term and through ``\mathrm{Re}`` and ``\dot m`` in ``K`` and on the right. With ``\phi = h`` it is the head
equation of [Sommers2018](@cite). In an unfilled or
uplifted cell, ``N`` is fixed at ``p_o`` or ``0`` and the water depth takes up the balance instead:
the ``\partial W / \partial t`` that the gap does not carry stays on the left as a storage term in
``h`` (``W - b = h - z_b`` or ``h - h_o``). The head equation and the gap equation are coupled through
``b`` in ``K`` and through ``N`` in the closure; [Discretization](@ref) shows how a time step solves them.

## Boundary conditions

Every cell has a mask value. The equations above hold on `GROUNDED` cells. The other values set the
boundary conditions:

| Mask | Condition |
|:-----|:----------|
| `OCEAN` | ``h = z_b + \left(p_\mathrm{atm} - \rho_{sw} g \min(z_b, 0)\right) / (\rho_w g)``: the pressure of the ocean at the bed |
| `LAND` | ``h = z_b + p_\mathrm{atm} / (\rho_w g)``: water leaves at atmospheric pressure |
| `OTHER_BASIN`, `FROZEN_BED` | no flux across the face; the head is held. On a frozen bed ``p_w = 0`` and ``N = p_o`` |
| edge of the grid | no flux |

By default `OCEAN` and `LAND` faces are outflow only (`outflow_only_land`, `outflow_only_ocean`): the
face closes when the boundary head is above the water level ``\phi`` of the grounded cell, so the
boundary takes water from the bed but does not feed it.

## Parameters

The default values of `ModelParameters`:

| Symbol | Keyword | Default | Meaning |
|:-------|:--------|:--------|:--------|
| ``\rho_w``, ``\rho_i``, ``\rho_{sw}`` | `rho_w`, `rho_i`, `rho_sw` | 1000, 910, 1027 kg m⁻³ | densities of water, ice and sea water |
| ``g`` | `g` | 9.81 m s⁻² | gravity |
| ``\nu`` | `nu` | 1.787 × 10⁻⁶ m² s⁻¹ | kinematic viscosity of water |
| ``\omega`` | `omega` | 10⁻³ | laminar–turbulent transition [Sommers2018](@cite) |
| ``n`` | `n` | 3 | Glen's law exponent |
| ``L`` | `L` | 3.34 × 10⁵ J kg⁻¹ | latent heat of fusion |
| ``b_r``, ``l_r`` | `br`, `lr` | 0.1 m, 2 m | bump height and spacing |
| ``b_c`` | `b_c` | 0 (``l_c = b``) | creep cutoff [Felden2023](@cite) |
| ``c_t``, ``c_w`` | `ct`, `cw` | 7.5 × 10⁻⁸ K Pa⁻¹, 4220 J kg⁻¹ K⁻¹ | change of melting point with pressure, heat capacity of water |
| ``e_v`` | `e_v` | 0 | englacial void ratio |
| ``p_\mathrm{atm}`` | `p_atm` | 0 Pa | pressure at `LAND` and of the sea surface |
| ``b_{min}``, ``b_\max`` | `b_min`, `b_max` | 10⁻⁶ m, ∞ | bounds on ``b`` |
| ``N_{min}``, ``N_\max`` | `N_min`, `N_max` | −∞, ∞ | bounds on ``N`` |

The fields ``z_b``, ``z_s``, ``\mathbf{u}_b``, ``A``, ``G``, ``q_T`` and ``i_{eb}`` are inputs, set in the
`State` (step 4 of the tutorial) or by a coupled ice-flow model.
