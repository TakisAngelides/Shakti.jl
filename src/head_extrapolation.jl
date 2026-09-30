"""
$(TYPEDSIGNATURES)

Initial guess for each timestep's head solve, extrapolated in time from the last converged heads
(held by [`Simulation`](@ref) as `he`, applied by [`step!`](@ref) before the head solve).
Multiple dispatch picks [`NoHeadExtrapolation`](@ref) (start from the previous step's head, the
original behaviour) or [`HeadExtrapolation`](@ref).
"""
abstract type AbstractHeadExtrapolation end

"""
$(TYPEDSIGNATURES)

No extrapolation: each head solve starts from the previous step's converged head.
"""
struct NoHeadExtrapolation <: AbstractHeadExtrapolation end

"""
$(TYPEDSIGNATURES)

Polynomial extrapolation in time of the head, of degree `order` (1 = linear, 2 = quadratic, ...),
through the last `order + 1` converged heads, evaluated at the new step's end time -- a better
starting point for the Picard/Newton iteration than the previous head alone, which needs fewer
iterations to reach the same tolerance (the answer itself is unchanged: only the starting point
moves).

# Notes

- Only converged heads enter the history. The initial condition is not a solution of the head
  equation, so the first extrapolation happens at the third step (linear) -- until enough history
  exists, the highest degree the available heads allow is used, down to none. A step that fails to
  converge clears the history, so a bad head is never extrapolated from.
- Weights are Lagrange weights on the actual step end times, so variable `dt`
  ([`AdaptiveTimeStep`](@ref)) is handled exactly.
- Only `GROUNDED` cells are extrapolated: Dirichlet (`OCEAN`/`LAND`) heads are fixed, and
  `OTHER_BASIN`/`FROZEN_BED` rows hold their current value, which extrapolation would make drift.
- Memory: `order + 1` head-sized arrays. Cost per step: `order + 1` fused broadcasts.
- Not used under [`ParabolicHeadScheme`](@ref): its storage term needs the unextrapolated head of
  the previous step as its `h_old`, which it reads from `s.h` at the start of the solve.
"""
mutable struct HeadExtrapolation{A <: AbstractArray} <: AbstractHeadExtrapolation
    order::Int
    hist::Vector{A}  # converged heads, newest first (hist[1] = the last converged head)
    times::Vector{Float64} # the model time each was converged at, same order (Float64 even under Float32 fields: model times reach ~1e9 s)
    count::Int       # how many entries of hist/times are valid
    clock::Float64   # own model clock, advanced by each recorded step's dt: step! can be driven without run! (which is what advances sim.total_time), so total_time can't be relied on here
end

"""
$(TYPEDSIGNATURES)

Builds a [`HeadExtrapolation`](@ref) of degree `order` (>= 1) on grid `g`.
"""
function HeadExtrapolation(g::Grid{F}; order::Int = 1) where F
    order >= 1 || error("HeadExtrapolation: order must be >= 1 (got $order); use NoHeadExtrapolation() to disable")
    hist = [initialize_center_field(g) for _ in 1:order+1]
    return HeadExtrapolation(order, hist, zeros(Float64, order + 1), 0, 0.0)
end

"""
$(TYPEDSIGNATURES)

Forgets all stored heads (no-op under [`NoHeadExtrapolation`](@ref)).
"""
reset_head_history!(::NoHeadExtrapolation) = nothing
reset_head_history!(he::HeadExtrapolation) = (he.count = 0; he.clock = 0.0; nothing)

"""
$(TYPEDSIGNATURES)

Records `s.h` as the newest converged head, at the end of a step of length `dt` -- or, if the step
did not converge, clears the history instead.
"""
record_head!(::NoHeadExtrapolation, s::State, dt, converged::Bool) = nothing
function record_head!(he::HeadExtrapolation, s::State, dt, converged::Bool)
    he.clock += Float64(dt)
    if !converged
        he.count = 0
        return nothing
    end
    oldest = pop!(he.hist) # recycle the oldest buffer as the newest entry, no allocation
    copyto!(oldest, s.h)
    pushfirst!(he.hist, oldest)
    for j in length(he.times):-1:2
        he.times[j] = he.times[j-1]
    end
    he.times[1] = he.clock
    he.count = min(he.count + 1, he.order + 1)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Overwrites the `GROUNDED` cells of `s.h` with the extrapolation to the end of the coming step of
length `dt`; returns the degree actually used (`0` = nothing done, too little history).
"""
extrapolate_head!(::NoHeadExtrapolation, s::State, dt) = 0
function extrapolate_head!(he::HeadExtrapolation, s::State, dt)
    npts = he.count
    npts < 2 && return 0
    ts = he.times
    t_new = he.clock + Float64(dt)
    for j in 2:npts # a zero-length step (dt = 0) would put two points at the same time
        ts[j] < ts[j-1] || return 0
    end
    grounded = GROUNDED
    for j in 1:npts
        # Lagrange weight of point j at t_new, times taken relative to the newest point
        w64 = 1.0
        for m in 1:npts
            m == j && continue
            w64 *= (t_new - ts[m]) / (ts[j] - ts[m])
        end
        w = eltype(s.h)(w64)
        hj = he.hist[j]
        if j == 1
            @. s.h = ifelse(s.mask == grounded, w * hj, s.h)
        else
            @. s.h = ifelse(s.mask == grounded, s.h + w * hj, s.h)
        end
    end
    return npts - 1
end
