# Shakti.jl

Shakti.jl is a Julia implementation of the transient subglacial hydrology model SHAKTI
[Sommers2018](@cite). On a regular grid it evolves the water-filled gap between ice and bed and gives
the hydraulic head, the water flux and the effective pressure at the ice base. One set of equations
covers distributed (laminar) and channelized (turbulent) drainage.

By default the sheet follows GlaDS-2 [Wells2026](@cite): cavities can be partly empty and water
above the overburden lifts the ice, so `0 ≤ p_w ≤ p_o`. SHAKTI's original sheet model can be
selected with keywords to `Simulation`, shown in [the tutorial](@ref "7. The sheet") and run in the
[Helheim](@ref Helheim) and [Drang Drung](@ref DrangDrung) examples. Channel-wall diffusion from
SUHMO [Felden2023](@cite) is an option.

The aim is a Shakti fast enough to be coupled to an ice-flow model such as
[Yelmo](https://github.com/fesmc/yelmo) over long runs, on CPU and GPU.

## Defaults

| Choice | Default | Opt-out |
|---:|---:|---:|
| `cavity_filling` | `UnfilledCavities()` | `FilledCavities()` |
| `uplift` | `IceUplift()` | `NoUplift()` |
| `hydraulic_potential` | `SheetTopPotential()` | `BedPotential()` |
| `k_face_choice` | `"implicit_upwind"` | `"upwind"` (short `dt` only) |
| `gap_coupling` | `ConsistentGapCoupling()` | `ExplicitGapCoupling()` |
| `b_max` | `Inf` | a finite cap |

`"implicit_upwind"` needs `LUDirectSolver` or `CUDSSDirectSolver(grid)`.
SHAKTI's original equations are `FilledCavities()`, `NoUplift()`, `BedPotential()` and
`ExplicitGapCoupling()` together.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/TakisAngelides/Shakti.jl")
```

## Where next

* [Tutorial](@ref): every option, on a synthetic ice-sheet margin
* Examples: [Helheim](@ref Helheim), [Drang Drung](@ref DrangDrung), [Thwaites](@ref Thwaites),
  [Greenland](@ref Greenland), [Antarctica](@ref Antarctica)
* [Seasonal melt](@ref SeasonalMeltInput), [Parabolic head scheme](@ref ParabolicScheme),
  [SUHMO channel-wall diffusion](@ref SUHMO), [Performance](@ref)
* [API Reference](@ref)
