# Shakti.jl

Shakti.jl is a Julia implementation of the transient subglacial hydrology model Shakti
[Sommers2018](@cite). On a regular grid it evolves the water-filled gap between ice and bed and gives
the hydraulic head, the water flux and the effective pressure at the ice base. One set of equations
covers distributed (laminar) and channelized (turbulent) drainage.

By default the physics represented by the code follows GlaDS-2 [Wells2026](@cite), where cavities can be partly empty and water
above the overburden lifts the ice, so `0 ≤ p_w ≤ p_o`. Shakti's original model can be
selected with keywords to `Simulation`, shown in [the tutorial](@ref "7. The sheet") and run in the
[Helheim](@ref Helheim) and [Drang Drung](@ref DrangDrung) examples. Channel-wall diffusion from
Suhmo [Felden2023](@cite) is also implemented in the code as an option.

The aim is a Shakti fast enough to be coupled to an ice-flow model such as
[Yelmo](https://github.com/fesmc/yelmo) over long runs, on CPU and GPU.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/TakisAngelides/Shakti.jl")
```

## Where next

* [Tutorial](@ref): every option the user can make is explained here using a synthetic ice-sheet margin.
* Examples of simulations with real topographies: [Helheim](@ref Helheim), [Drang Drung](@ref DrangDrung), [Thwaites](@ref Thwaites),
  [Greenland](@ref Greenland), [Antarctica](@ref Antarctica)
* Example for seasonal melt forcing: [Seasonal melt](@ref SeasonalMeltInput)
* Example on the englacial storage term: [Parabolic head scheme](@ref ParabolicScheme),
* Example on how to activate the Suhmo channel-wall diffusion term: [Suhmo channel-wall diffusion](@ref SUHMO)
* Example on the gap coupling and the closed water budget: [Gap coupling](@ref GapCoupling)
* Timing of performance for different setups: [Performance](@ref)
* [API Reference](@ref)
