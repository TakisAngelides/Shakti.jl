"""
$(TYPEDSIGNATURES)

How `s.ieb` (englacial-to-bed meltwater input, e.g. moulins/crevasses) is set up and evolved over
time -- multiple dispatch on the concrete subtype picks a fixed field ([`ConstantMeltInput`](@ref)),
a time-varying seasonal cycle ([`SeasonalMeltInput`](@ref)), or a spatially-localized,
time-ramped point source ([`GaussianMoulinMeltInput`](@ref)). Every subtype implements
[`update_ieb!`](@ref) (called once per real timestep, outside the Picard loop, so `s.ieb` stays
fixed across every Picard iteration within that timestep). `s.ieb` itself is seeded directly (`s.ieb
.= ieb`) during [`set_initial_conditions!`](@ref) and read directly (`ieb[i, j]`) inside
`linear_solver.jl`'s `@parallel` assembly kernels -- neither of those needs to dispatch on the
concrete `AbstractMeltInput` subtype.
"""
abstract type AbstractMeltInput end

"""
$(TYPEDSIGNATURES)

A time-independent `ieb` field: whatever was seeded into `state.ieb` during
[`set_initial_conditions!`](@ref) is used unchanged for the whole run ([`update_ieb!`](@ref) is a
no-op).
"""
struct ConstantMeltInput <: AbstractMeltInput end

"""
$(TYPEDSIGNATURES)

No-op: `ieb` never changes after [`set_initial_conditions!`](@ref) under
[`ConstantMeltInput`](@ref), so there's nothing to do per timestep.
"""
@inline update_ieb!(::ConstantMeltInput, state::State, t) = state

"""
$(TYPEDSIGNATURES)

Reproduces the seasonal-cycle experiment from the original SHAKTI paper (Sommers et al. 2018,
Sect. 3.3): `i_e->b` is applied uniformly over the whole domain, held at a winter baseline
(`i_min`) except during a cosine-shaped melt-season window `[t_start, t_start+period]` (year
fraction), where it swings up to a summer peak and back down to `i_min` at both ends of the
window.

# Notes

Holds only scalar fields (no arrays): like [`ConstantMeltInput`](@ref)/[`Arithmetic`](@ref)/
[`Harmonic`](@ref), this struct is passed by value into `linear_solver.jl`'s `@parallel` kernels,
which requires kernel arguments to be bitstypes -- an array field would break that. `omega`
(`2*pi/period`) is precomputed once at construction rather than recomputed every
[`update_ieb!`](@ref) call, same idiom as `model_parameters.jl`'s `canonical_exponent`.
"""
struct SeasonalMeltInput{F <: AbstractFloat} <: AbstractMeltInput
    t_start::F           # start of the melt-season window, year fraction (0-1)
    period::F            # window width in years (t_end - t_start)
    omega::F             # 2*pi/period
    amplitude::F         # cosine amplitude, m a^-1
    offset::F            # cosine vertical offset, m a^-1
    i_min::F             # baseline input outside the window, m a^-1
    seconds_per_year::F
end

"""
$(TYPEDSIGNATURES)

Builds a [`SeasonalMeltInput`](@ref) from keyword arguments (all have defaults matching the
original SHAKTI paper's Sect. 3.3 experiment), converting every value to `floattype` and
precomputing `period`/`omega`.
"""
function SeasonalMeltInput(; t_start = 0.4, t_end = 0.7, amplitude = 492.75, offset = 493.75, i_min = 1.0, seconds_per_year = 365 * 86400.0)

    F = floattype
    period = t_end - t_start
    return SeasonalMeltInput(F(t_start), F(period), F(2 * pi / period), F(amplitude), F(offset), F(i_min), F(seconds_per_year))

end

"""
$(TYPEDSIGNATURES)

Updates `state.ieb` (uniformly over the whole domain) for the current simulation time `t`
(elapsed seconds, see `run.jl`'s `total_time`), following `mi`'s cosine-shaped seasonal cycle.

# Notes

Called once per real timestep, outside the Picard loop, so `state.ieb` stays fixed across every
Picard iteration within that timestep -- same as [`ConstantMeltInput`](@ref)'s `ieb` being fixed
for the whole run. The `t` input is elapsed simulation time in seconds.
"""
@inline function update_ieb!(mi::SeasonalMeltInput, state::State, t)
    F  = eltype(state.ieb)
    yf = mod(t / mi.seconds_per_year, one(F)) # year fraction, wraps for multi-year runs
    i_ma = (mi.t_start <= yf <= mi.t_start + mi.period) ? # if we are within the window of cosine input
           (mi.offset - mi.amplitude * cos(mi.omega * (yf - mi.t_start))) : # give this cosine input
            mi.i_min # otherwise give this minimum background input only
    state.ieb .= i_ma / mi.seconds_per_year # uniform over the whole domain; m a^-1 -> m s^-1
    return state
end

"""
$(TYPEDSIGNATURES)

A single point source (moulin) with a fixed 2D Gaussian spatial footprint and a linear temporal
ramp from `0` at `t=0` up to its full discharge `Q_max` at `t=ramp_duration`, held constant
thereafter -- reproduces Felden et al. (2023, SUHMO)'s Sect. 4.2 channelized convergence test
case ("the moulin source term follows a spatial Gaussian profile... the moulin input is gradually
increased in time, from 0 at time t=0s to the maximum value after about a month"). The paper
doesn't give the Gaussian's exact width or the ramp's exact functional shape -- `sigma` and the
choice of a plain linear ramp (vs. e.g. a smoothstep) are this port's own, reasonable but
unverified-against-the-paper choices.

# Notes

Unlike [`SeasonalMeltInput`](@ref), this DOES hold a full `(Nx, Ny)` array field (`shape`) -- fine
here since `mi` itself is never passed into a `@parallel` kernel (only [`update_ieb!`](@ref)'s own
plain broadcast reads it, same as how [`LinearSlidingLaw`](@ref)'s `C` field holds a full array).
`shape` is precomputed once at construction, normalized so `sum(shape)*dx*dy ≈ 1` (an
un-truncated Gaussian integrates to exactly 1 over the whole plane; on a finite domain this is
only approximate, good enough as long as `sigma` is small relative to the domain size, as it is
here).
"""
struct GaussianMoulinMeltInput{A <: AbstractArray, F <: AbstractFloat} <: AbstractMeltInput
    shape::A          # precomputed, normalized spatial Gaussian (m^-2), sum(shape)*dx*dy ≈ 1
    Q_max::F          # peak discharge, m^3 s^-1
    ramp_duration::F  # seconds to reach Q_max from a t=0 start; held at Q_max thereafter
end

"""
$(TYPEDSIGNATURES)

Builds a [`GaussianMoulinMeltInput`](@ref) centered at `(x0, y0)` (grid coordinates, same origin
as `grid.x`/`grid.y`) with spatial standard deviation `sigma`, ramping up to `Q_max` over
`ramp_duration` seconds.
"""
function GaussianMoulinMeltInput(grid::Grid, x0, y0, sigma, Q_max, ramp_duration)
    F = floattype
    shape = [exp(-((xi - x0)^2 + (yi - y0)^2) / (2 * sigma^2)) for xi in grid.x, yi in grid.y]
    shape ./= sum(shape) * grid.dx * grid.dy
    return GaussianMoulinMeltInput(F.(shape), F(Q_max), F(ramp_duration))
end

"""
$(TYPEDSIGNATURES)

Updates `state.ieb` to `mi.shape * Q_max * min(t/ramp_duration, 1)` -- the fixed spatial Gaussian
footprint scaled by the current point on the linear ramp (elapsed simulation time `t`, seconds).
"""
@inline function update_ieb!(mi::GaussianMoulinMeltInput, state::State, t)
    F = eltype(state.ieb)
    ramp = min(t / mi.ramp_duration, one(F))
    state.ieb .= mi.shape .* mi.Q_max .* ramp
    return state
end
