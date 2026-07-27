"""
$(TYPEDSIGNATURES)

How `s.ieb` (englacial-to-bed meltwater input, e.g. moulins/crevasses) is set up and evolved over
time -- multiple dispatch on the concrete subtype picks a fixed field ([`ConstantMeltInput`](@ref))
or a time-varying seasonal cycle ([`SeasonalMeltInput`](@ref)). Every subtype implements
[`initialize_ieb!`](@ref) (called once, during [`set_initial_conditions!`](@ref)),
[`compute_ieb!`](@ref) (a single-cell lookup, called from inside `linear_solver.jl`'s `@parallel`
assembly kernels), and [`update_ieb!`](@ref) (called once per real timestep, outside the Picard
loop, so `s.ieb` stays fixed across every Picard iteration within that timestep).
"""
abstract type AbstractMeltInput end

"""
$(TYPEDSIGNATURES)

A time-independent `ieb` field: whatever was passed to [`initialize_ieb!`](@ref) is used
unchanged for the whole run ([`update_ieb!`](@ref) is a no-op).
"""
struct ConstantMeltInput <: AbstractMeltInput end

"""
$(TYPEDSIGNATURES)

Seeds `state.ieb` from `ieb`. Called once, during [`set_initial_conditions!`](@ref).
"""
function initialize_ieb!(::ConstantMeltInput, state::State, ieb::AbstractArray)
    state.ieb .= ieb
end

"""
$(TYPEDSIGNATURES)

Single-cell lookup of the current `ieb` value at `(i, j)`, used inside `linear_solver.jl`'s
`@parallel` assembly kernels (which need a bitstype-safe way to read melt input without carrying
a whole `AbstractMeltInput` struct into the kernel).
"""
function compute_ieb!(::ConstantMeltInput, ieb::AbstractArray, i::Int, j::Int)
    return ieb[i, j]
end

"""
$(TYPEDSIGNATURES)

No-op: `ieb` never changes after [`initialize_ieb!`](@ref) under [`ConstantMeltInput`](@ref), so
there's nothing to do per timestep.
"""
update_ieb!(::ConstantMeltInput, state::State, t) = state

"""
$(TYPEDSIGNATURES)

Reproduces the seasonal-cycle experiment from the original SHAKTI paper (Sommers et al. 2018,
Sect. 3.3): `i_e->b` is applied uniformly over the whole domain, held at a winter baseline
(`i_min`) except during a cosine-shaped melt-season window `[t_start, t_start+period]` (year
fraction), where it swings up to a summer peak and back down to `i_min` at both ends of the
window -- continuously, since the paper's default `amplitude`/`offset` (`492.75`/`493.75`) make
the cosine's boundary value exactly `i_min`.

# Notes

Holds only scalar fields (no arrays): like [`ConstantMeltInput`](@ref)/[`Arithmetic`](@ref)/
[`Harmonic`](@ref), this struct is passed by value into `linear_solver.jl`'s `@parallel` kernels
(for [`compute_ieb!`](@ref)'s dispatch), which requires kernel arguments to be bitstypes -- an
array field would break that. `omega` (`2*pi/period`) is precomputed once at construction rather
than recomputed every [`update_ieb!`](@ref) call, same idiom as `model_parameters.jl`'s
`canonical_exponent`.
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
function SeasonalMeltInput(;
    t_start = 0.4, t_end = 0.7, amplitude = 492.75, offset = 493.75, i_min = 1.0, seconds_per_year = 365 * 86400.0)

    F = floattype
    period = t_end - t_start
    return SeasonalMeltInput(
        F(t_start), F(period), F(2 * pi / period), F(amplitude), F(offset), F(i_min), F(seconds_per_year),
    )
end

"""
$(TYPEDSIGNATURES)

Seeds `state.ieb` from `ieb`. Called once, during [`set_initial_conditions!`](@ref); the actual
seasonal cycle is applied later, per timestep, by [`update_ieb!`](@ref).
"""
function initialize_ieb!(::SeasonalMeltInput, state::State, ieb::AbstractArray)
    state.ieb .= ieb
end

"""
$(TYPEDSIGNATURES)

Single-cell lookup of the current `ieb` value at `(i, j)` (already up to date for this timestep,
via [`update_ieb!`](@ref)), used inside `linear_solver.jl`'s `@parallel` assembly kernels.
"""
function compute_ieb!(::SeasonalMeltInput, ieb::AbstractArray, i::Int, j::Int)
    return ieb[i, j]
end

"""
$(TYPEDSIGNATURES)

Updates `state.ieb` (uniformly over the whole domain) for the current simulation time `t`
(elapsed seconds, see `run.jl`'s `total_time`), following `mi`'s cosine-shaped seasonal cycle.

# Notes

Called once per real timestep, outside the Picard loop, so `state.ieb` stays fixed across every
Picard iteration within that timestep -- same as [`ConstantMeltInput`](@ref)'s `ieb` being fixed
for the whole run.
"""
function update_ieb!(mi::SeasonalMeltInput, state::State, t)
    F  = eltype(state.ieb)
    yf = mod(t / mi.seconds_per_year, one(F)) # year fraction, wraps for multi-year runs
    i_ma = (mi.t_start <= yf <= mi.t_start + mi.period) ?
        (mi.offset - mi.amplitude * cos(mi.omega * (yf - mi.t_start))) :
        mi.i_min
    state.ieb .= i_ma / mi.seconds_per_year # uniform over the whole domain; m a^-1 -> m s^-1
    return state
end
