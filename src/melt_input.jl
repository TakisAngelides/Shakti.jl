abstract type AbstractMeltInput end

struct ConstantMeltInput <: AbstractMeltInput end

function initialize_ieb!(::ConstantMeltInput, state::State, ieb::AbstractArray)
    state.ieb .= ieb
end

function compute_ieb!(::ConstantMeltInput, ieb::AbstractArray, i::Int, j::Int)
    return ieb[i, j]
end

# ieb never changes after initialize_ieb!, so there's nothing to do per timestep.
update_ieb!(::ConstantMeltInput, state::State, t) = state

# Reproduces the seasonal-cycle experiment from the original SHAKTI paper
# (Sommers et al. 2018, Sect. 3.3): i_e->b is applied uniformly over the
# whole domain, held at a winter baseline (i_min) except during a
# cosine-shaped melt-season window [t_start, t_start+period] (year fraction),
# where it swings up to a summer peak and back down to i_min at both ends of
# the window -- continuously, since the paper's default amplitude/offset
# (492.75/493.75) make the cosine's boundary value exactly i_min. Holds only
# scalar fields (no arrays): like ConstantMeltInput/Arithmetic/Harmonic, this
# struct is passed by value into linear_solver.jl's @parallel kernels (for
# compute_ieb!'s dispatch), which requires kernel arguments to be bitstypes
# -- an array field would break that. `omega` (2*pi/period) is precomputed
# once at construction rather than recomputed every update_ieb! call, same
# idiom as model_parameters.jl's canonical_exponent.
struct SeasonalMeltInput{F <: AbstractFloat} <: AbstractMeltInput
    t_start::F           # start of the melt-season window, year fraction (0-1)
    period::F            # window width in years (t_end - t_start)
    omega::F             # 2*pi/period
    amplitude::F         # cosine amplitude, m a^-1
    offset::F            # cosine vertical offset, m a^-1
    i_min::F             # baseline input outside the window, m a^-1
    seconds_per_year::F
end

function SeasonalMeltInput(;
    t_start = 0.4, t_end = 0.7, amplitude = 492.75, offset = 493.75, i_min = 1.0, seconds_per_year = 365 * 86400.0)

    F = floattype
    period = t_end - t_start
    return SeasonalMeltInput(
        F(t_start), F(period), F(2 * pi / period), F(amplitude), F(offset), F(i_min), F(seconds_per_year),
    )
end

function initialize_ieb!(::SeasonalMeltInput, state::State, ieb::AbstractArray)
    state.ieb .= ieb
end

function compute_ieb!(::SeasonalMeltInput, ieb::AbstractArray, i::Int, j::Int)
    return ieb[i, j]
end

# t is the simulation's elapsed time in seconds (see run.jl's total_time).
# Called once per real timestep, outside the Picard loop, so state.ieb stays
# fixed across every Picard iteration within that timestep, same as
# ConstantMeltInput's ieb is fixed for the whole run.
function update_ieb!(mi::SeasonalMeltInput, state::State, t)
    F  = eltype(state.ieb)
    yf = mod(t / mi.seconds_per_year, one(F)) # year fraction, wraps for multi-year runs
    i_ma = (mi.t_start <= yf <= mi.t_start + mi.period) ?
        (mi.offset - mi.amplitude * cos(mi.omega * (yf - mi.t_start))) :
        mi.i_min
    state.ieb .= i_ma / mi.seconds_per_year # uniform over the whole domain; m a^-1 -> m s^-1
    return state
end
