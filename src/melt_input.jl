"""
$(TYPEDSIGNATURES)

How `s.ieb` (englacial-to-bed meltwater input, e.g. moulins/crevasses) is set up and evolved over
time -- multiple dispatch on the concrete subtype picks a fixed field ([`ConstantMeltInput`](@ref))
or a time-varying seasonal cycle ([`SeasonalMeltInput`](@ref)). Every subtype implements
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
