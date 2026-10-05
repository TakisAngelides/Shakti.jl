# =============================================================================
# Gap height (b) evolution
# =============================================================================
# Runs once per time step, after the Picard loop has converged on h. compute_beta!,
# compute_lc!, and compute_b_x!/compute_b_y! are then refreshed from the new b, ready
# to be read by the next time step's Picard loop (compute_q_x!/compute_q_y!
# read b_x/b_y; the next compute_b! call reads beta; the elliptic/parabolic RHS
# assembly kernels, linear_solver.jl, read lc).
#
# This file is included before simulation.jl (same reason melt_rate.jl is
# included before elliptic_solver.jl, see its own docstring note):
# Simulation's struct definition needs AbstractOpenBySlidingScheme/
# AbstractCreepLengthScheme in scope to type-annotate `oss`/`cls`. The
# Simulation-dispatching compute_b!(sim::Simulation) family (mirroring
# compute_b! dispatching on sim.gs) therefore lives in run.jl instead, next to
# step_b! -- their sole caller -- rather than here.

"""
$(TYPEDSIGNATURES)

Whether `compute_beta!` includes the opening-by-sliding term -- multiple dispatch on the concrete
subtype ([`WithOpenBySliding`](@ref)/[`NoOpenBySliding`](@ref)), decided ONCE in `Simulation`'s
constructor (from `p.br`, see `simulation.jl`) rather than every call, so turning it off skips
`compute_beta_kernel!`'s kernel launch entirely instead of launching it to compute a term that's
mathematically always zero when `p.br == 0` -- same "dispatch on a type decided once outside the
hot loop" idiom as [`MeltTerms`](@ref) (`melt_rate.jl`).
"""
abstract type AbstractOpenBySlidingScheme end

"""
$(TYPEDSIGNATURES)

Include the opening-by-sliding term (`compute_beta_kernel!`) in [`compute_beta!`](@ref).
"""
struct WithOpenBySliding <: AbstractOpenBySlidingScheme end

"""
$(TYPEDSIGNATURES)

Skip the opening-by-sliding term in [`compute_beta!`](@ref) entirely (not just compute a term
that's identically zero): `s.beta` is left untouched, which is correct since `p.br == 0` is what
selects this scheme in the first place, and `max(0, (0 - b)/lr) == 0` for any `b >= 0` anyway.
"""
struct NoOpenBySliding <: AbstractOpenBySlidingScheme end

# Opening rate of the gap by sliding over bedrock bumps of height br and
# spacing lr (Rothlisberger-style cavity opening): positive only while the
# gap is still smaller than the bump height, zero once b has grown past br.
@parallel_indices (ix, iy) function compute_beta_kernel!(beta, b, br, lr)
    if ix <= size(beta, 1) && iy <= size(beta, 2)
        beta[ix, iy] = max(zero(br), (br - b[ix, iy]) / lr)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.beta` (opening rate of the gap by sliding over bedrock bumps of height `p.br` and
spacing `p.lr`, Röthlisberger-style cavity opening): positive only while the gap is still smaller
than the bump height, zero once `b` has grown past `br`.

Dispatches on `oss` (decided once, from `p.br`, see [`AbstractOpenBySlidingScheme`](@ref)): the
[`NoOpenBySliding`](@ref) method skips the kernel launch entirely, since `p.br == 0` makes the term
identically zero mathematically.
"""
compute_beta!(s::State, p::ModelParameters, ::WithOpenBySliding) = (@parallel compute_beta_kernel!(s.beta, s.b, p.br, p.lr); s)

"""
$(TYPEDSIGNATURES)

`compute_beta!` without the opening-by-sliding term -- see the `WithOpenBySliding` method's
docstring above.
"""
compute_beta!(s::State, p::ModelParameters, ::NoOpenBySliding) = s

"""
$(TYPEDSIGNATURES)

Which ice-creep length scale `l_c` [`compute_lc!`](@ref) writes into `s.lc` -- multiple dispatch on
the concrete subtype ([`StandardCreep`](@ref)/[`CreepCutoff`](@ref)) picks between today's `l_c = b`
and the piecewise cutoff of Felden et al. (2023, Eq. 7), decided ONCE in `Simulation`'s constructor
(from `p.b_c`, see `simulation.jl`) -- same "dispatch on a type decided once outside the hot loop"
idiom as [`AbstractOpenBySlidingScheme`](@ref)/[`MeltTerms`](@ref). `s.lc` (not `s.b` directly) is
what the creep-closure term reads everywhere downstream: `compute_b_implicit_kernel!`/
`compute_b_explicit_kernel!`/`compute_b_fully_implicit_kernel!` (this file, which additionally need
`cls`/`p.b_c` directly since they solve for the *new* `b`, so can't just read the lagged `s.lc`) and
the elliptic/parabolic RHS assembly kernels (`linear_solver.jl`, which read the lagged `s.lc`
exactly like they read `s.beta` -- both refreshed once per timestep by `step_b!`, ready for the
next timestep's Picard loop).
"""
abstract type AbstractCreepLengthScheme end

"""
$(TYPEDSIGNATURES)

Today's behavior: `l_c = b`, unconditionally. Selected automatically when `p.b_c == 0`.
"""
struct StandardCreep <: AbstractCreepLengthScheme end

"""
$(TYPEDSIGNATURES)

Felden et al. (2023, SUHMO, https://doi.org/10.5194/gmd-16-407-2023) Eq. 7's creep length scale:
`l_c = b*(1 - (b_c-b)/b_c) = b^2/b_c` for `b <= p.b_c`, else `l_c = b` (continuous at `b = p.b_c`).
Below the cutoff, ice-creep closure (`A|N|^(n-1)*N*l_c`) scales as `b^2` instead of `b`, cutting it
off faster than the standard linear term as `b -> 0` -- intended to let sheet-like drainage survive
in places where the linear closure would otherwise close it out entirely. Selected automatically
when `p.b_c != 0`.

# Known instability -- NOT safe to enable without checking for near-flotation cells first

Real-dataset testing (2026-09-16, real-dataset SUHMO test pass; reconfirmed 2026-09-17 against
this exact code via a shortened reproduction, both in project memory/`suhmo.tex`) found
`CreepCutoff` triggers an unbounded gap-height runaway at cells where `N` goes deeply negative
while `b` is still below `b_c`. On Helheim (`b_c=0.05`, `b_max=Inf`), one cell reached
`b=70.66m`/`K=8.1e10` by the end of a 90-day run (baseline: `b=0.27m`) -- and the 2026-09-17
check reproduced the same divergence at the same cell (`b~3.9m` by day 30) from a fresh run,
confirming this is not an artifact of one earlier run.

**Mechanism** (`implicit_creep_update`, this file): while `b <= b_c` and `N < 0` (so the closure
coefficient `C = A|N|^(n-1)*N` is negative), this scheme's cutoff branch returns
`b_below = b_old + dt*opening` directly -- i.e. it DROPS the `(1 + dt*C)` implicit-closure
denominator entirely in that regime, unlike [`StandardCreep`](@ref) which always applies it (even
if only mildly, since `|dt*C|` is normally small). With nothing damping it, `b` then grows by the
bare opening term every step, unchecked, until it crosses `b_c` onto the ordinary linear branch --
by which point the growth already has enough momentum that it doesn't turn over.

**Not a generic problem**: clean (1 Picard iteration/step, no instability) on 3 of 5 real datasets
tested (Drang Drung, Thwaites, Pan-Antarctica) at the same or larger `b_c`. Failed only on the two
datasets with known persistent near-flotation cells -- Helheim (uncontained runaway) and Greenland
(contained non-convergence, ~1% of steps, no NaN/Inf). The datasets that stayed clean already use
[`CellNClamping`](@ref) at their own known-unstable cells (Thwaites, Pan-Antarctica), for reasons
predating this branch -- Helheim does not, and has no finite `p.b_max` either.

**Before enabling `CreepCutoff` (`p.b_c != 0`) on a new dataset**: check for cells with a
persistent or deep negative-`N` excursion and consider [`CellNClamping`](@ref) there, and set a
finite `p.b_max`. Neither is a confirmed complete fix -- Greenland had `CellNClamping` and still
saw contained non-convergence -- so treat this as a real, open risk to check for per-dataset, not a
solved problem. See `test/helheim/diagnose_creepcutoff_runaway.jl` for the original root-cause
trace.
"""
struct CreepCutoff <: AbstractCreepLengthScheme end

"""
$(TYPEDSIGNATURES)

`l_c(b)` per [`AbstractCreepLengthScheme`](@ref)'s dispatch -- `@inline`d into every kernel that
uses it (`compute_lc_kernel!` below, plus `compute_b_implicit_kernel!`/`compute_b_explicit_kernel!`/
`compute_dt_rate_kernel!`, which all need the *current* `b` rather than the lagged `s.lc`).
"""
@inline creep_length(::StandardCreep, b, b_c) = b
@inline creep_length(::CreepCutoff, b, b_c) = b <= b_c ? b * (one(b) - (b_c - b) / b_c) : b

"""
$(TYPEDSIGNATURES)

`dl_c/db`, the local slope of [`creep_length`](@ref) -- feeds [`compute_dt_rate_kernel!`](@ref)'s
closure-term relaxation rate (`C*dl_c/db`, the correct generalization of `C` once closure is no
longer linear in `b`).
"""
@inline creep_length_slope(::StandardCreep, b, b_c) = one(b)
@inline creep_length_slope(::CreepCutoff, b, b_c) = b <= b_c ? 2*b/b_c : one(b)

@parallel_indices (ix, iy) function compute_lc_kernel!(lc, b, b_c, cls::AbstractCreepLengthScheme)
    if ix <= size(lc, 1) && iy <= size(lc, 2)
        lc[ix, iy] = creep_length(cls, b[ix, iy], b_c)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.lc` (the ice-creep length scale read by the creep-closure term everywhere downstream,
see [`AbstractCreepLengthScheme`](@ref)) from the current `s.b`. Called once per timestep by
[`step_b!`](@ref), right after `s.b` is updated -- same timing as [`compute_beta!`](@ref), so `s.lc`
stays lagged (not the *new* `b`) for exactly the same reason `s.beta` does.
"""
compute_lc!(s::State, p::ModelParameters, cls::AbstractCreepLengthScheme) = (@parallel compute_lc_kernel!(s.lc, s.b, p.b_c, cls); s)

@parallel_indices (ix, iy) function compute_b_x_kernel!(b_x, b)
    nx1 = size(b_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(b_x, 2)
        if ix == 1 # boundary face takes the value of the nearest center value
            b_x[ix, iy] = b[1, iy]
        elseif ix == nx1 # boundary face takes the value of the nearest center value
            b_x[ix, iy] = b[nx1-1, iy]
        else # face value is taken to be the arithmetic average of the center values touching that face
            b_x[ix, iy] = (b[ix, iy] + b[ix-1, iy]) / 2
        end
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.b_x` by staggering `s.b` onto x-faces (boundary faces duplicate the nearest cell,
interior faces average their two neighbours).
"""
compute_b_x!(s::State) = (@parallel compute_b_x_kernel!(s.b_x, s.b); s)

@parallel_indices (ix, iy) function compute_b_y_kernel!(b_y, b)
    ny1 = size(b_y, 2) # ny + 1
    if ix <= size(b_y, 1) && iy <= ny1
        if iy == 1
            b_y[ix, iy] = b[ix, 1]
        elseif iy == ny1
            b_y[ix, iy] = b[ix, ny1-1]
        else
            b_y[ix, iy] = (b[ix, iy] + b[ix, iy-1]) / 2
        end
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.b_y` by staggering `s.b` onto y-faces (boundary faces duplicate the nearest cell,
interior faces average their two neighbours).
"""
compute_b_y!(s::State) = (@parallel compute_b_y_kernel!(s.b_y, s.b); s)

"""
$(TYPEDSIGNATURES)

One step of the linear relaxation ODE `db/dt = S - R*b` over `dt`, from `b0`. `R >= 0` (closure
and/or opening-by-sliding damping the gap): backward Euler, `(b0 + dt*S)/(1 + dt*R)` -- the
formula every gap scheme has always used, unchanged. `R < 0` (`N < 0`, so the "closure" term
`C*b` is really an opening term, and the equation describes genuine exponential growth):
backward Euler would divide by `1 - dt*|R|`, blowing up at `dt = 1/|R|` and turning `b` negative
beyond it, so the exact solution with `S`/`R` frozen over the step is used instead,
`b0*e^{|R|dt} + S*(e^{|R|dt} - 1)/|R|` (via `expm1`, accurate as `|R|*dt -> 0`) -- finite and
non-negative for every `dt`, so neither branch imposes a time-step limit.
"""
@inline function relax_update(b0, S, R, dt)
    R >= zero(R) && return (b0 + dt * S) / (1 + dt * R)
    x = -R * dt
    em = expm1(x)
    return b0 + em * b0 + S * dt * (em / x)
end

"""
$(TYPEDSIGNATURES)

Closed-form backward-Euler update of the closure-only ODE `b_{k+1} = b_k + dt*(opening -
C*l_c(b_{k+1}))`, dispatched on `cls` (see [`AbstractCreepLengthScheme`](@ref)):
[`StandardCreep`](@ref) is the plain linear solve `compute_b_implicit_kernel!` always used before
this generalization; [`CreepCutoff`](@ref) additionally has a self-consistent closed form despite
`l_c` being quadratic (not linear) in `b` below `p.b_c` -- same "solve both branches, keep whichever
is self-consistent" idea as [`compute_b_fully_implicit_kernel!`](@ref)'s `beta` branches, just for
`l_c`'s own piecewise definition instead:

  - Branch "`b_{k+1} <= b_c`" (`l_c = b_{k+1}^2/b_c`): `b_{k+1} + dt*C/b_c*b_{k+1}^2 = b_k +
    dt*opening` is a quadratic in `b_{k+1}` with a unique non-negative root (quadratic formula,
    positive branch).
  - Branch "`b_{k+1} > b_c`" (`l_c = b_{k+1}`): reduces to the same linear solve as
    [`StandardCreep`](@ref).
"""
@inline implicit_creep_update(::StandardCreep, b_old, opening, C, dt, b_c) = relax_update(b_old, opening, C, dt)

@inline function implicit_creep_update(::CreepCutoff, b_old, opening, C, dt, b_c)
    rhs = b_old + dt * opening
    a = dt * C / b_c
    # a<=0 (C<=0, i.e. N<=0): the quadratic branch's own derivation (l_c=b^2/b_c) assumes C>0
    # (an actually-closing force) -- it stops making physical sense once the term is opening
    # instead. Fall back to the StandardCreep-equivalent form (same formula the b_below>b_c branch
    # below already uses) rather than skipping the (1+dt*C) closure-feedback term entirely, which
    # is what caused a real, confirmed runaway (see CreepCutoff's own docstring/project notes).
    # (relax_update: backward Euler for C >= 0, the exact exponential solution for C < 0.)
    b_below = a > 0 ? (-one(a) + sqrt(max(zero(a), one(a) + 4 * a * rhs))) / (2 * a) : relax_update(b_old, opening, C, dt)
    if b_below <= b_c
        return b_below
    else # branch 1's own assumption (b_{k+1} <= b_c) failed -- l_c(b_{k+1}) is actually b_{k+1}
        return relax_update(b_old, opening, C, dt)
    end
end

@parallel_indices (ix, iy) function compute_b_implicit_kernel!(b, mask, mdot, beta, abs_ub, A_visc, N, rho_i, n_minus_1, dt, b_min, b_max, cls::AbstractCreepLengthScheme, b_c)
    if ix <= size(b, 1) && iy <= size(b, 2) && mask[ix, iy] == GROUNDED # we only evolve the water thickness if the cell has grounded ice
        C = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]
        opening = mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy]
        b[ix, iy] = clamp( # b_min/b_max bound the water thickness for numerical stability -- b_max specifically guards against a runaway b->K->q->mdot->b feedback (creep closure, which depends on N, vanishes as N->0, so nothing bounds b from above there without this; see ModelParameters' own docstring)
            implicit_creep_update(cls, b[ix, iy], opening, C, dt, b_c),
            b_min, b_max)
    end
    return
end

@parallel_indices (ix, iy) function compute_b_explicit_kernel!(b, mask, mdot, beta, abs_ub, A_visc, N, rho_i, n_minus_1, dt, b_min, b_max, cls::AbstractCreepLengthScheme, b_c)
    if ix <= size(b, 1) && iy <= size(b, 2) && mask[ix, iy] == GROUNDED
        b[ix, iy] = clamp(
            b[ix, iy] + dt * (mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy] -
                A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * creep_length(cls, b[ix, iy], b_c)),
            b_min, b_max)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Closed-form fully-implicit solve of `b_{k+1} = b_k + dt*(mdot/rho_i + beta(b_{k+1})*|u_b| -
C*l_c(b_{k+1}))`, dispatched on `cls` (see [`AbstractCreepLengthScheme`](@ref)). Both `beta(b) =
max(0, (br-b)/lr)` and (under [`CreepCutoff`](@ref)) `l_c(b)` are piecewise functions of the same
unknown `b_{k+1}`, each switching at its own threshold (`br`/`b_c`, generally unrelated values) --
solving both simultaneously means checking every combination of "which side of `br`" x "which side
of `b_c`" the true root falls on, same "solve a branch, keep it only if self-consistent with its
own assumption" idea as [`compute_b_implicit_kernel!`](@ref)'s `l_c` branches, just with twice as
many branches to try.

Writing `opening0 = b_k + dt*mdot/rho_i` (the part common to every branch) and `gamma = |u_b|/lr`:

  - `b_{k+1} < br` and `b_{k+1} <= b_c` (both active): substituting both piecewise pieces gives a
    quadratic, `(dt*C/b_c)*b_{k+1}^2 + (1+dt*gamma)*b_{k+1} - (opening0 + dt*gamma*br) = 0`.
  - `b_{k+1} < br` and `b_{k+1} > b_c` (`beta` active, `l_c` linear): linear,
    `b_{k+1} = (opening0 + dt*gamma*br) / (1 + dt*(gamma+C))`.
  - `b_{k+1} >= br` and `b_{k+1} <= b_c` (`beta` zero, `l_c` cutoff): quadratic,
    `(dt*C/b_c)*b_{k+1}^2 + b_{k+1} - opening0 = 0` -- the same equation
    [`compute_b_implicit_kernel!`](@ref)'s `CreepCutoff` branch solves, just with a plain
    `mdot/rho_i` opening rather than a lagged-`beta` one.
  - `b_{k+1} >= br` and `b_{k+1} > b_c` (both zero/linear): `b_{k+1} = opening0/(1+dt*C)`, the same
    closure-only solve [`compute_b_implicit_kernel!`](@ref) always falls back to.

`br` and `b_c` needn't be ordered a particular way for this to work: whichever is bigger, one of the
four combinations above becomes geometrically impossible on the `b_{k+1}` axis, and that branch's
own candidate simply never passes its self-consistency check -- so trying all four in a fixed order
and returning the first self-consistent one is correct either way, at the cost of at most one extra
(cheap) candidate evaluation per cell. Under [`StandardCreep`](@ref) this collapses back to the
original two branches (`l_c` linear everywhere, so only the `beta`-threshold matters).

Every linear branch goes through [`relax_update`](@ref): backward Euler while its rate (`C` or
`gamma + C`) is non-negative, and the exact exponential solution when it is negative (`N < 0`,
where "closure" is really opening) -- so no branch ever divides by a vanishing or negative
`1 + dt*rate`. The quadratic branches only run for `C > 0`. Unconditionally stable for any `dt`,
unlike [`compute_b_implicit_kernel!`](@ref) (which inherits a
`dt` cap from evaluating `beta` at the lagged `b`; see its own docstring).
"""
# `b0` is the old gap height and `m` the melt-opening rate mdot/rho_i (opening0 = b0 + dt*m); they
# are passed separately because relax_update's exponential branch needs them apart.
@inline function fully_implicit_creep_update(::StandardCreep, b0, m, gamma, C, dt, br, b_c)
    b_below = relax_update(b0, m + gamma * br, gamma + C, dt)
    return b_below < br ? b_below : relax_update(b0, m, C, dt)
end

# Original (opening0, ...) form, kept for existing callers: identical to the above whenever the
# rates are non-negative (backward Euler only ever sees b0 + dt*m).
@inline fully_implicit_creep_update(cls::AbstractCreepLengthScheme, opening0, gamma, C, dt, br, b_c) =
    fully_implicit_creep_update(cls, opening0, zero(opening0), gamma, C, dt, br, b_c)

@inline function fully_implicit_creep_update(::CreepCutoff, b0, m, gamma, C, dt, br, b_c)
    opening0 = b0 + dt * m
    a = dt * C / b_c

    # Branch (i): beta active (b_{k+1} < br) AND l_c cutoff (b_{k+1} <= b_c)
    beta_coef = 1 + dt * gamma
    rhs_i = opening0 + dt * gamma * br
    b1 = a > 0 ? (-beta_coef + sqrt(max(zero(a), beta_coef^2 + 4 * a * rhs_i))) / (2 * a) : relax_update(b0, m + gamma * br, gamma + C, dt)
    if b1 < br && b1 <= b_c
        return b1
    end

    # Branch (ii): beta active (b_{k+1} < br) AND l_c linear (b_{k+1} > b_c)
    b2 = relax_update(b0, m + gamma * br, gamma + C, dt)
    if b2 < br && b2 > b_c
        return b2
    end

    # Branch (iii): beta zero (b_{k+1} >= br) AND l_c cutoff (b_{k+1} <= b_c)
    b3 = a > 0 ? (-one(a) + sqrt(max(zero(a), one(a) + 4 * a * opening0))) / (2 * a) : relax_update(b0, m, C, dt)
    if b3 >= br && b3 <= b_c
        return b3
    end

    # Branch (iv): beta zero (b_{k+1} >= br) AND l_c linear (b_{k+1} > b_c) -- always
    # self-consistent as the final fallback (exactly one of the four branches must be, since
    # beta/l_c are continuous and together partition the b_{k+1} axis into non-overlapping pieces).
    return relax_update(b0, m, C, dt)
end

@parallel_indices (ix, iy) function compute_b_fully_implicit_kernel!(b, mask, mdot, abs_ub, A_visc, N, rho_i, n_minus_1, dt, b_min, b_max, br, lr, cls::AbstractCreepLengthScheme, b_c)
    if ix <= size(b, 1) && iy <= size(b, 2) && mask[ix, iy] == GROUNDED
        C = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]
        gamma = abs_ub[ix, iy] / lr
        b[ix, iy] = clamp(fully_implicit_creep_update(cls, b[ix, iy], mdot[ix, iy] / rho_i, gamma, C, dt, br, b_c), b_min, b_max)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Writes `C + gamma` (the same local relaxation rate behind [`compute_b_fully_implicit_kernel!`](@ref)'s
branch-1 denominator, i.e. the reciprocal of the physical timescale `tau` the timestep-selection
report is built on) into `rate` for every `GROUNDED` cell; non-grounded cells get `0`, which is
harmless for a `maximum` reduction (never wins) and is explicitly excluded from a percentile
reduction via `AdaptiveTimeStep`'s precomputed `grounded_indices`. `gamma` only contributes while
`b < br` (mirrors `compute_b_fully_implicit_kernel!`'s own branch condition) and is `0` outright
when `br == 0` (opening-by-sliding off) -- so a domain that never uses it (e.g. Drang Drung) isn't
penalized with an artificially small `dt`. Under [`CreepCutoff`](@ref), `C` alone is no longer the
closure term's local slope (since `l_c` is quadratic, not linear, in `b` below `p.b_c`) -- it's
scaled by [`creep_length_slope`](@ref) (`dl_c/db`, `1` under [`StandardCreep`](@ref) so this is a
no-op there) to keep the rate/timescale estimate correct.
"""
@parallel_indices (ix, iy) function compute_dt_rate_kernel!(rate, mask, b, abs_ub, A_visc, N, n_minus_1, br, lr, cls::AbstractCreepLengthScheme, b_c)
    if ix <= size(rate, 1) && iy <= size(rate, 2)
        if mask[ix, iy] == GROUNDED
            C = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * creep_length_slope(cls, b[ix, iy], b_c)
            gamma = (br > 0 && b[ix, iy] < br) ? abs_ub[ix, iy] / lr : zero(C)
            rate[ix, iy] = C + gamma
        else
            rate[ix, iy] = zero(eltype(rate))
        end
    end
    return
end


"""
$(TYPEDSIGNATURES)

Freeze-on capacity of one cell [m/s of ice]: the largest freezing rate `f` (melt opening
`mdot/rho_i = -f`) the gap equation can take this step without the gap falling below `b_min`,

    C = max(0, (b - b_min)/dt + beta(b_beta)*|u_b| - A|N|^(n-1)*N*l_c(b_creep)),

i.e. room in the gap above the floor spread over the step, plus new room opened by sliding, minus
room creep closure uses up anyway. `b_beta`/`b_creep` are where the gap scheme evaluates the two
terms: `b_min` for an implicit term (the end-of-step gap, which at the limit is `b_min`), the old `b`
for an explicit or lagged one. Exact for the backward-Euler branches of the gap update; with `N < 0`
the update relaxes exponentially instead and this is the first-order estimate.
"""
@inline function freeze_on_capacity_cell(b, abs_ub, Cn, dt, b_min, br, lr, cls::AbstractCreepLengthScheme, b_c, b_beta, b_creep)
    beta = br > 0 ? max(zero(b), (br - b_beta) / lr) : zero(b)
    return max(zero(b), (b - b_min) / dt + beta * abs_ub - Cn * creep_length(cls, b_creep, b_c))
end

"""
$(TYPEDSIGNATURES)

[`freeze_on_capacity_cell`](@ref) for the local terms of the SUHMO gap update under
[`WithDiffusion`](@ref) ([`b_diffusion_local_terms`](@ref)), ignoring the diffusion itself: creep
with `C > 0` is implicit through the lagged ratio `lc/b` (so it closes `C*(lc/b)*b_min` at the
limit), creep with `C <= 0` is explicit (`C*lc`), and opening by sliding is implicit while the lagged
`beta > 0` (`gamma*(br - b_min)`). Setting the solved `b` to `b_min` with `mdot/rho_i = -f` gives
`f` exactly when `D = 0`.
"""
@inline function freeze_on_capacity_cell_diffusion(b, beta, abs_ub, Cn, lc, dt, b_min, br, lr)
    creep   = Cn > zero(Cn) ? Cn * (b > zero(b) ? lc / b : one(b)) * b_min : Cn * lc
    sliding = beta > zero(beta) ? (abs_ub / lr) * (br - b_min) : zero(b)
    return max(zero(b), (b - b_min) / dt + sliding - creep)
end

# Freezing turns water into ice: at rate `f` (ice thickness per time) it consumes `f*rho_i/rho_w` of water per
# bed area, so a cell holding little water (a nearly empty cavity, see UnfilledCavities) cannot take more
# than its water stock `W = b - b_empty` per host step: f <= (rho_w/rho_i)*W/dt. Only applied when
# `water_limited` (unfilled cavities on); with a water-filled gap (W = b) the gap room above is the limit.
@parallel_indices (ix, iy) function freeze_on_capacity_kernel!(Cap, mask, b, beta, lc, abs_ub, A_visc, N, n_minus_1, dt, b_min, br, lr, cls::AbstractCreepLengthScheme, b_c, beta_at_bmin, creep_at_bmin, diffusion, b_empty, water_ratio, water_limited)
    if ix <= size(Cap, 1) && iy <= size(Cap, 2)
        if mask[ix, iy] == GROUNDED
            bb = b[ix, iy]
            Cn = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]
            Cap[ix, iy] = diffusion ?
                freeze_on_capacity_cell_diffusion(bb, beta[ix, iy], abs_ub[ix, iy], Cn, lc[ix, iy], dt, b_min, br, lr) :
                freeze_on_capacity_cell(bb, abs_ub[ix, iy], Cn, dt, b_min, br, lr, cls, b_c,
                                        beta_at_bmin ? b_min : bb, creep_at_bmin ? b_min : bb)
            if water_limited
                Cap[ix, iy] = min(Cap[ix, iy], max(zero(bb), bb - b_empty[ix, iy]) * water_ratio / dt)
            end
        else
            Cap[ix, iy] = zero(eltype(Cap))
        end
    end
    return
end
