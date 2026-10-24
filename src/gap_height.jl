# =============================================================================
# Gap height (b) evolution
# =============================================================================
# Runs once per time step, after the Picard loop has converged on h. compute_beta! and
# compute_lc! are then refreshed from the new b, ready to be read by the next time step
# (the next compute_b! call reads beta; the head assembly kernels,
# linear_solver.jl, read lc). compute_b_x!/compute_b_y! are refreshed too, but the Picard
# loop no longer reads b_x/b_y (the face conductances take the water depth b_w,
# compute_face_flux!); they are kept as face diagnostics.
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
# The ice base is b + b_uplift above the bed; the bumps it still rides over are those above b_uplift, of height
# bump = max(0, br - b_uplift) (= br without uplift, bit for bit), and sliding opens the gap only while b < bump.
# b_uplift is the water W beyond the gap where the ice is lifted (stor_up > 0 and h above the overburden head), else 0.
@inline uplift_layer(W, b, h, zb, po, stor_up, rho_wg) = stor_up > 0 && h > zb + po / rho_wg ? max(zero(W), W - b) : zero(W)
@parallel_indices (ix, iy) function compute_beta_kernel!(beta, bump, b, W, h, zb, po, stor_up, rho_wg, br, lr)
    if ix <= size(beta, 1) && iy <= size(beta, 2)
        bu = max(zero(br), br - uplift_layer(W[ix, iy], b[ix, iy], h[ix, iy], zb[ix, iy], po[ix, iy], stor_up[ix, iy], rho_wg))
        bump[ix, iy] = bu
        beta[ix, iy] = max(zero(br), (bu - b[ix, iy]) / lr)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.beta` (opening rate of the gap by sliding over bedrock bumps of height `p.br` and
spacing `p.lr`, Röthlisberger-style cavity opening): positive only while the gap is still smaller
than the bump height, zero once `b` has grown past `br`. Under [`IceUplift`](@ref) the ice base is
`b + b_uplift` above the bed, `b_uplift = W - b` the water beyond the gap where the head is above the overburden
(`s.W`, the water), so the bumps it still rides over have height `s.bump = max(0, br - b_uplift)`
(also stored, for the fully implicit gap scheme and the gap update the head rows assume) and
`beta = max(0, (bump - b)/lr)`: ice floating above the bumps opens no cavity. Without uplift `bump = br`.

Dispatches on `oss` (decided once, from `p.br`, see [`AbstractOpenBySlidingScheme`](@ref)): the
[`NoOpenBySliding`](@ref) method skips the kernel launch entirely, since `p.br == 0` makes the term
identically zero mathematically.
"""
compute_beta!(s::State, p::ModelParameters, ::WithOpenBySliding) =
    (@parallel compute_beta_kernel!(s.beta, s.bump, s.b, s.W, s.h, s.zb, s.po, s.stor_up, p.rho_w * p.g, p.br, p.lr); s)

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
the head assembly kernels (`linear_solver.jl`, which read the lagged `s.lc`
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
coefficient `C = A|N|^(n-1)*N` is negative), the cutoff branch used to return
`b_below = b_old + dt*opening` directly, dropping the `(1 + dt*C)` implicit-closure term in that
regime, unlike [`StandardCreep`](@ref). `b` then grew by the bare opening term every step until it
crossed `b_c` onto the linear branch, by which point the growth did not turn over. That branch now
falls back to the `StandardCreep` form at `C <= 0` (`relax_update`: the exact exponential at
`N < 0`); the real-data runs above predate that change and have not been repeated, so the caution
below stands.

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

`dl_c/db`, the local slope of `creep_length` -- feeds `compute_dt_rate_kernel!`'s
closure-term relaxation rate (`C*dl_c/db`, the correct generalization of `C` once closure is no
longer linear in `b`).
"""
@inline creep_length_slope(::StandardCreep, b, b_c) = one(b)
@inline creep_length_slope(::CreepCutoff, b, b_c) = b <= b_c ? 2*b/b_c : one(b)

# With the closure rate `C = A|N|^(n-1)N` known: at `C < 0` (`N < 0`) the "closure" opens the gap, and every scheme takes
# `l_c = b` there (the implicit updates' quadratic branch assumes `C > 0`; `C*b^2/b_c` growth would blow up in finite
# time). One definition, so the gap ODE at `N < 0` is the same under every gap scheme. At `C = 0` either form gives zero.
@inline creep_length(cls::AbstractCreepLengthScheme, b, b_c, C) = C < zero(C) ? b : creep_length(cls, b, b_c)
@inline creep_length_slope(cls::AbstractCreepLengthScheme, b, b_c, C) = C < zero(C) ? one(b) : creep_length_slope(cls, b, b_c)

@parallel_indices (ix, iy) function compute_lc_kernel!(lc, b, N, b_c, cls::AbstractCreepLengthScheme)
    if ix <= size(lc, 1) && iy <= size(lc, 2)
        lc[ix, iy] = creep_length(cls, b[ix, iy], b_c, N[ix, iy]) # sign of N is the sign of C
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.lc` (the ice-creep length scale read by the creep-closure term everywhere downstream,
see [`AbstractCreepLengthScheme`](@ref)) from the current `s.b`. Called once per timestep by
[`step_b!`](@ref), right after `s.b` is updated -- same timing as [`compute_beta!`](@ref), so `s.lc`
stays lagged (not the *new* `b`) for exactly the same reason `s.beta` does. Where `s.N < 0` it is `b` under
either scheme (see the four-argument `creep_length`).
"""
compute_lc!(s::State, p::ModelParameters, cls::AbstractCreepLengthScheme) = (@parallel compute_lc_kernel!(s.lc, s.b, s.N, p.b_c, cls); s)

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
is self-consistent" idea as `compute_b_fully_implicit_kernel!`'s `beta` branches, just for
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
        C = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]
        b[ix, iy] = clamp(
            b[ix, iy] + dt * (mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy] - C * creep_length(cls, b[ix, iy], b_c, C)),
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

Every linear branch goes through `relax_update`: backward Euler while its rate (`C` or
`gamma + C`) is non-negative, and the exact exponential solution when it is negative (`N < 0`,
where "closure" is really opening) -- so no branch ever divides by a vanishing or negative
`1 + dt*rate`. The quadratic branches only run for `C > 0`. Unconditionally stable for any `dt`,
unlike [`ImplicitGapScheme`](@ref) (which inherits a `dt` cap from evaluating `beta` at the lagged `b`; see its
docstring).
"""
# `b0` is the old gap height and `m` the melt-opening rate mdot/rho_i (opening0 = b0 + dt*m); they
# are passed separately because relax_update's exponential branch needs them apart.
#
# For C >= 0 both branches are backward Euler on a monotone map, so exactly one is self-consistent, and a rejected
# branch 1 leaves branch 2 >= br. For C < 0 (N < 0, only reachable without IceUplift) branch 2 is the exponential
# solution, which can land below br after branch 1 was rejected: the new gap then fell as the old one grew, a jump
# that ConsistentGapCoupling's dG/dN straddled. Flooring branch 2 at br keeps the update continuous and monotone in
# b0 and N, and changes nothing for C >= 0.
@inline function fully_implicit_creep_update(::StandardCreep, b0, m, gamma, C, dt, br, b_c)
    b_below = relax_update(b0, m + gamma * br, gamma + C, dt)
    return b_below < br ? b_below : max(relax_update(b0, m, C, dt), br)
end

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

    # Branch (iv): beta zero (b_{k+1} >= br) AND l_c linear (b_{k+1} > b_c). For C < 0 it is floored at br, as in the
    # StandardCreep update above.
    b4 = relax_update(b0, m, C, dt)
    C < zero(C) && return max(b4, br)
    b4 >= br && b4 > b_c && return b4
    # For C >= 0 exactly one branch is self-consistent in exact arithmetic (beta and l_c are continuous and partition the
    # b_{k+1} axis). When none passes, the solution sits on a kink and rounding put each candidate just across it: a
    # starting gap whose new gap is b_c gave b1 = b_c + 4e-14 and b2 = b_c - 2e-17, and branch (iv), returned unchecked,
    # 4e-6 m below b_c. The residual of the backward-Euler equation is increasing in b_{k+1}, so the kink where it is
    # smallest is the solution.
    resid(b) = abs(b - opening0 - dt * (gamma * max(br - b, zero(b)) - C * creep_length(CreepCutoff(), b, b_c)))
    return resid(b_c) <= resid(br) ? b_c : br
end

@parallel_indices (ix, iy) function compute_b_fully_implicit_kernel!(b, mask, mdot, abs_ub, A_visc, N, rho_i, n_minus_1, dt, b_min, b_max, bump, lr, cls::AbstractCreepLengthScheme, b_c)
    if ix <= size(b, 1) && iy <= size(b, 2) && mask[ix, iy] == GROUNDED
        C = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]
        gamma = abs_ub[ix, iy] / lr
        b[ix, iy] = clamp(fully_implicit_creep_update(cls, b[ix, iy], mdot[ix, iy] / rho_i, gamma, C, dt, bump[ix, iy], b_c), b_min, b_max) # bump: br less the uplift layer (compute_beta!)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Writes `C + gamma` (the same local relaxation rate behind `compute_b_fully_implicit_kernel!`'s
branch-1 denominator, i.e. the reciprocal of the physical timescale `tau` the timestep-selection
report is built on) into `rate` for every `GROUNDED` cell whose gap is above `b_min`; other cells get `-Inf`, which
never wins a `maximum` reduction and marks the cell for exclusion from a percentile reduction
([`rate_statistic`](@ref)), so both follow the mask as it is at this step. A gap at `b_min` is held there, so its rate
(fastest where thick ice closes it within minutes) says nothing about how the solution changes. `gamma` only contributes while
`b < br` (mirrors `compute_b_fully_implicit_kernel!`'s own branch condition) and is `0` outright
when `br == 0` (opening-by-sliding off) -- so a domain that never uses it (e.g. Drang Drung) isn't
penalized with an artificially small `dt`. Under [`CreepCutoff`](@ref), `C` alone is no longer the
closure term's local slope (since `l_c` is quadratic, not linear, in `b` below `p.b_c`) -- it's
scaled by [`creep_length_slope`](@ref) (`dl_c/db`, `1` under [`StandardCreep`](@ref) so this is a
no-op there) to keep the rate/timescale estimate correct.
"""
@parallel_indices (ix, iy) function compute_dt_rate_kernel!(rate, mask, b, abs_ub, A_visc, N, n_minus_1, bump, lr, cls::AbstractCreepLengthScheme, b_c, b_min)
    if ix <= size(rate, 1) && iy <= size(rate, 2)
        if mask[ix, iy] == GROUNDED && b[ix, iy] > b_min
            Cn = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]
            C = Cn * creep_length_slope(cls, b[ix, iy], b_c, Cn)
            br = bump[ix, iy] # bump height under the ice base (compute_beta!)
            gamma = (br > 0 && b[ix, iy] < br) ? abs_ub[ix, iy] / lr : zero(C)
            rate[ix, iy] = C + gamma
        else
            rate[ix, iy] = typemin(eltype(rate)) # -Inf: not rated (rate_statistic)
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
    return max(zero(b), (b - b_min) / dt + beta * abs_ub - Cn * creep_length(cls, b_creep, b_c, Cn))
end

"""
$(TYPEDSIGNATURES)

`freeze_on_capacity_cell` for the local terms of the SUHMO gap update under
[`WithDiffusion`](@ref) (`b_diffusion_local_terms`), ignoring the diffusion itself: creep
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
# than its water stock `W` (State.W) per host step: f <= (rho_w/rho_i)*W/dt. Only applied when
# `water_limited` (unfilled cavities on); with a water-filled gap (W = b) the gap room above is the limit.
@parallel_indices (ix, iy) function freeze_on_capacity_kernel!(Cap, mask, b, beta, lc, abs_ub, A_visc, N, n_minus_1, dt, b_min, bump, lr, cls::AbstractCreepLengthScheme, b_c, beta_at_bmin, creep_at_bmin, diffusion, W, water_ratio, water_limited)
    if ix <= size(Cap, 1) && iy <= size(Cap, 2)
        if mask[ix, iy] == GROUNDED
            bb = b[ix, iy]
            Cn = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]
            br = bump[ix, iy] # bump height under the ice base (compute_beta!)
            Cap[ix, iy] = diffusion ?
                freeze_on_capacity_cell_diffusion(bb, beta[ix, iy], abs_ub[ix, iy], Cn, lc[ix, iy], dt, b_min, br, lr) :
                freeze_on_capacity_cell(bb, abs_ub[ix, iy], Cn, dt, b_min, br, lr, cls, b_c,
                                        beta_at_bmin ? b_min : bb, creep_at_bmin ? b_min : bb)
            if water_limited
                Cap[ix, iy] = min(Cap[ix, iy], max(zero(bb), W[ix, iy]) * water_ratio / dt)
            end
        else
            Cap[ix, iy] = zero(eltype(Cap))
        end
    end
    return
end

# Freeze-on capacity under the dry floor (freeze_on_capacity!, run.jl): the room is the water W (State.W) plus the opening
# by sliding (and minus the explicit creep closure under ExplicitGapScheme; the implicit schemes cannot close the water below
# zero), and freezing takes no more water than W.
@parallel_indices (ix, iy) function freeze_on_capacity_dry_floor_kernel!(Cap, mask, Wo, beta, abs_ub, A_visc, N, n_minus_1, dt, cls::AbstractCreepLengthScheme, b_c, explicit_creep, water_ratio)
    if ix <= size(Cap, 1) && iy <= size(Cap, 2)
        if mask[ix, iy] == GROUNDED
            W = Wo[ix, iy]
            Cn = A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]
            creep = explicit_creep ? Cn * creep_length(cls, W, b_c, Cn) : zero(W)
            Cap[ix, iy] = max(zero(W), min(W / dt + beta[ix, iy] * abs_ub[ix, iy] - creep, W * water_ratio / dt))
        else
            Cap[ix, iy] = zero(eltype(Cap))
        end
    end
    return
end
