"""
$(TYPEDSIGNATURES)

How the hydraulic head is advanced each timestep -- multiple dispatch on the concrete subtype (see [`step_h!`](@ref) in
`run.jl`). [`PicardHeadScheme`](@ref) is the one there is.
"""
abstract type AbstractHeadScheme end

"""
$(TYPEDSIGNATURES)

Head scheme of every [`Simulation`](@ref): each timestep, the [`PicardSolver`](@ref) `ps` is run to convergence
([`Picard_loop!`](@ref)). The head equation is elliptic for `p.e_v == 0` and parabolic otherwise: the rows then also store
englacial water, `e_v (h - zb)` (Sommers et al. 2018 Eq. 13), backward Euler from the head at the start of the step.
"""
struct PicardHeadScheme{PS <: PicardSolver} <: AbstractHeadScheme
    ps::PS
end

"""
$(TYPEDSIGNATURES)

How the gap height `b` is time-integrated each timestep -- multiple dispatch on the concrete
subtype picks between [`ExplicitGapScheme`](@ref), [`ImplicitGapScheme`](@ref), and
[`FullyImplicitGapScheme`](@ref) (see `compute_b!` in `gap_height.jl`).
"""
abstract type AbstractGapScheme end

"""
$(TYPEDSIGNATURES)

Explicit (forward-Euler) update of `b`: cheaper per step, but only stable for small enough `dt`.
"""
struct ExplicitGapScheme <: AbstractGapScheme end

"""
$(TYPEDSIGNATURES)

Implicit (backward-Euler) update of `b`: the creep-closure term is unconditionally stable at any
`dt`, but the opening-by-sliding term (`beta(b)*|u_b|`, active when `p.br != 0`) is still evaluated
at the *lagged* `b` (last step's value, via `s.beta`) rather than the new one -- so this scheme
inherits a real, if usually generous, `dt` cap from that term whenever `p.br != 0` and `b < p.br`
(`beta = (br - b)/lr`, so the explicit opening overshoots `br` once `dt*|u_b|/lr > 1`). Confirmed in practice,
not just in theory: a real AIS 32km run hit this at `dt=900s` and blew `b` up to the `b_max` safety
cap in one cell, something [`FullyImplicitGapScheme`](@ref) -- now the default -- doesn't do.
Unconditionally stable outright when `p.br == 0`. **Deprecated**: [`FullyImplicitGapScheme`](@ref) has the
same cost without that cap; selecting it warns, and the scheme will be removed.
"""
struct ImplicitGapScheme <: AbstractGapScheme end

"""
$(TYPEDSIGNATURES)

Fully implicit update of `b`: both the creep-closure term AND the opening-by-sliding term are
evaluated at the new (`k+1`) `b`, not the lagged one -- unconditionally stable for any `dt`,
regardless of `p.br`. `beta(b) = max(0, (p.br-b)/p.lr)` is piecewise-linear (not smooth, but not
genuinely nonlinear either), so this doesn't need a Newton iteration: `compute_b_fully_implicit_kernel!`
solves both linear branches (`b_{k+1} < p.br` and `b_{k+1} >= p.br`) in closed form and picks
whichever is self-consistent with its own assumption. Same per-step cost as
[`ImplicitGapScheme`](@ref) (often cheaper in practice, since Picard convergence tends to benefit
from not carrying the spurious runaway/clamp events `ImplicitGapScheme` is prone to) -- no extra
iteration. **The default choice** -- confirmed against `ImplicitGapScheme` and `ExplicitGapScheme`
in a 6-dt/2-domain (AIS/GrIS) sweep: all three agree closely at the finest `dt` (<0.3% relative
RMSE), and `FullyImplicitGapScheme` tracks `ImplicitGapScheme` far more tightly than
`ExplicitGapScheme` does as `dt` grows coarser.
"""
struct FullyImplicitGapScheme <: AbstractGapScheme end

"""
$(TYPEDSIGNATURES)

How `sim.dt[]` is chosen each timestep -- multiple dispatch on the concrete subtype picks between
[`FixedTimeStep`](@ref) (today's behavior: `dt` set once at construction, never changes) and
[`AdaptiveTimeStep`](@ref) (chosen every step by how the head solve converges).
"""
abstract type AbstractTimeStepScheme end

"""
$(TYPEDSIGNATURES)

`sim.dt[]` is set once at construction and never changes. Today's behavior; the default.
"""
struct FixedTimeStep <: AbstractTimeStepScheme end

"""
$(TYPEDSIGNATURES)

Chooses `sim.dt[]` every step by how the head solve converges, between `dt_min` and `dt_max`, and stops [`run!`](@ref)
at `target_time`. Under the default sheet the gap update is fully implicit and unconditionally stable, so what limits
the step is the Picard iteration at regime changes (cells filling, emptying or lifting the ice), not a relaxation
rate of the gap:

- a step whose head solve does not converge is rejected: the head is reset to the start of the step and the step is
  retried at half the `dt`, down to `dt_min` (a step at `dt_min` is kept whether it converged or not);
- after `grow_after` steps in a row that converged within `fast_iters` iterations, `dt` grows by the factor `grow`.

The step that `dt` starts from is the `Simulation`'s own `dt`. Only the head solve is redone: the gap, the water `W`,
the start-of-step melt and the head history change only once a step is accepted.

Optionally `dt` is also capped by the local relaxation rate of the gap equation,
`C + gamma = A_visc*|N|^(n-1)*N + |u_b|/lr` ([`compute_dt_rate_kernel!`](@ref)), over the cells whose gap is above
`b_min` (a gap at `b_min` is held there: creep closes it within minutes under thick ice and nothing changes):
`dt <= safety_factor / stat`, where `stat` is the maximum of that rate (`UsePercentile=false`, a single GPU-native
reduction) or a chosen percentile of it (`UsePercentile=true`, a sort and a host transfer each step, less swayed by one
outlier cell). This is an accuracy cap only; it is off unless `safety_factor` is given. `UsePercentile` is a type
parameter, so the choice is two `rate_statistic` methods rather than a branch in the per-step reduction. The rate is
taken from the previous step's converged `N`/`u_b`, the same lagged-coefficient idiom as `s.beta` in
[`ImplicitGapScheme`](@ref).

Build with the [`AdaptiveTimeStep`](@ref) convenience constructor below, not this one directly -- it sizes
`rate_field`/`host_rate`/`scratch` correctly.
"""
struct AdaptiveTimeStep{UsePercentile, F} <: AbstractTimeStepScheme
    safety_factor::F              # the rate cap safety_factor/stat; 0 = no cap
    percentile::F                 # quantile of the RATE field C+gamma (not of tau=1/(C+gamma)); unused when UsePercentile=false
    dt_min::F
    dt_max::F
    target_time::F                # physical duration to run for; run! breaks out of its tsteps loop once total_time[] reaches this
    grow::F                       # factor dt grows by after grow_after fast steps
    fast_iters::Int               # a step that converged within this many Picard iterations is fast
    grow_after::Int               # fast steps in a row before dt grows
    dt_ctrl::Base.RefValue{F}     # the step the controller asks for (NaN until the first step takes the Simulation's dt)
    fast_count::Base.RefValue{Int} # fast steps in a row since dt last grew or a step was rejected
    rejected::Base.RefValue{Int}  # rejected steps in this run (a diagnostic)
    rate_field::AbstractMatrix{F} # preallocated once (same backend as State), holds C+gamma per cell each step
    host_rate::Vector{F}          # preallocated once, CPU-side; empty when UsePercentile=false. Bridges a possibly-GPU-resident rate_field via one bulk copyto! (CUDA.jl/Metal.jl's real device->host transfer -- scalar-indexing a GPU array per grounded cell would be illegal/catastrophically slow, see checkpoint.jl's own note on this)
    scratch::Vector{F}            # preallocated once (one slot per cell); empty when UsePercentile=false. Each step gathers the rated cells (those compute_dt_rate_kernel! did not mark -Inf, so the mask as it is now) and sorts them in place
end

"""
$(TYPEDSIGNATURES)

Builds an [`AdaptiveTimeStep`](@ref), sizing its scratch buffers from `grid`/`state.mask`.

# Arguments
- `dt_min`/`dt_max`: hard bounds on `dt`. `dt_min` in particular guarantees [`run!`](@ref)'s loop terminates: total
  simulated time advances by at least `dt_min` every step (except a last step that lands on `target_time`), so a
  `target_time`-based run takes at most `ceil((target_time-total_time)/dt_min)` steps -- pass that same bound as
  `tsteps` when constructing the [`Simulation`](@ref). Rejected steps are retried within the same `tsteps` step.
- `target_time`: the physical duration (seconds) this run is meant to cover -- the final step is clipped to land exactly
  on it rather than overshoot, and the step before it is shortened when the remainder would otherwise be less than
  `dt_min` (see `update_dt!`).
- `grow`, `fast_iters`, `grow_after`: `dt` grows by `grow` after `grow_after` steps in a row whose head solve converged
  within `fast_iters` Picard iterations.
- `safety_factor`: `nothing` (default) for no rate cap; otherwise `dt <= safety_factor / stat` from the creep and
  sliding rate of the gap, e.g. `0.3` for the accuracy-oriented guidance worked out empirically for AIS/GrIS/Drang Drung.
- `use_percentile`: `false` (default) caps by the maximum of `C+gamma` (cheap, GPU-native, fully conservative); `true`
  by its `percentile`-th quantile instead (needs a sort; less swayed by one outlier cell).
- `percentile`: only used when `use_percentile=true`, in RATE space (not `tau` space) -- `0.75` means "the value such
  that 75% of the rated cells have `C+gamma` below it", i.e. the *fast-relaxing* end of the distribution.
"""
function AdaptiveTimeStep(grid::Grid, state::State; dt_min, dt_max, target_time, grow = 1.5, fast_iters::Int = 30, grow_after::Int = 3,
                          safety_factor = nothing, use_percentile::Bool = false, percentile = 0.75)
    F = eltype(state.N)
    grow > 1 || error("AdaptiveTimeStep: grow must be > 1")
    0 < dt_min <= dt_max || error("AdaptiveTimeStep: need 0 < dt_min <= dt_max")
    sf = safety_factor === nothing ? zero(F) : F(safety_factor)
    sf >= 0 || error("AdaptiveTimeStep: safety_factor must be positive")
    rate_field = initialize_center_field(grid)
    n = use_percentile ? length(rate_field) : 0 # every cell, so the rated set may change during the run (set_mask!, freezing)
    return AdaptiveTimeStep{use_percentile, F}(sf, F(percentile), F(dt_min), F(dt_max), F(target_time), F(grow), fast_iters, grow_after,
                                               Ref(F(NaN)), Ref(0), Ref(0), rate_field, Vector{F}(undef, n), Vector{F}(undef, n))
end

"""
$(TYPEDSIGNATURES)

The physical duration (seconds) `sim` should stop at, or `nothing` under [`FixedTimeStep`](@ref)
(meaning: run the full `sim.tsteps`, no early stop). Used by [`run!`](@ref)'s loop.
"""
target_time(::FixedTimeStep) = nothing
target_time(ts::AdaptiveTimeStep) = ts.target_time

"""
$(TYPEDSIGNATURES)

A sheet: the four choices that together make one physical model of the water at the bed, passed to
[`Simulation`](@ref) as `sheet`. Build one with [`GlaDS2Sheet`](@ref) (the default) or [`ShaktiSheet`](@ref); a
`Simulation` keyword (`cavity_filling`, `uplift`, `hydraulic_potential`, `gap_coupling`) overrides its field.
The face conductance is not part of the sheet: it follows from the cavity filling and uplift (`k_face_choice`).
"""
struct Sheet{CF <: AbstractCavityFilling, UP <: AbstractIceUplift, HP <: AbstractHydraulicPotential, GCP <: AbstractGapCoupling}
    cf::CF
    up::UP
    hp::HP
    gcp::GCP
end

"""
$(TYPEDSIGNATURES)

The sheet of GlaDS-2 (Wells et al. 2026), the default: [`UnfilledCavities`](@ref), [`IceUplift`](@ref),
[`SheetTopPotential`](@ref) and [`ConsistentGapCoupling`](@ref), with `"implicit_upwind"` faces. The pressure stays
in `0 <= p_w <= p_o`.
"""
GlaDS2Sheet() = Sheet(UnfilledCavities(), IceUplift(), SheetTopPotential(), ConsistentGapCoupling())

"""
$(TYPEDSIGNATURES)

SHAKTI's sheet (Sommers et al. 2018): [`FilledCavities`](@ref) and [`NoUplift`](@ref), with `"arithmetic"` faces, so
`p_w` may leave `[0, p_o]`. With `original = true` also [`BedPotential`](@ref) and [`ExplicitGapCoupling`](@ref): the
original equations bit for bit. Otherwise [`SheetTopPotential`](@ref) and [`ConsistentGapCoupling`](@ref).
"""
ShaktiSheet(; original::Bool = false) = original ? Sheet(FilledCavities(), NoUplift(), BedPotential(), ExplicitGapCoupling()) :
                                                   Sheet(FilledCavities(), NoUplift(), SheetTopPotential(), ConsistentGapCoupling())

# A face scheme by name (k_face_choice) or as itself.
k_face_scheme(kfs::AbstractKFaceScheme) = kfs
k_face_scheme(name::AbstractString) =
    name == "arithmetic"      ? Arithmetic() :
    name == "harmonic"        ? Harmonic() :
    name == "upwind"          ? Upwind() :
    name == "implicit_upwind" ? ImplicitUpwind() :
    error("Unknown k_face_choice: \"$name\" (expected \"arithmetic\", \"harmonic\", \"upwind\" or \"implicit_upwind\")")
k_face_name(kfs::AbstractKFaceScheme) = kfs isa ImplicitUpwind ? "implicit_upwind" : lowercase(string(nameof(typeof(kfs))))

"""
$(TYPEDSIGNATURES)

Everything needed to run a subglacial hydrology simulation: the grid, state, model parameters,
and every "which scheme/law" choice (head, gap, melt-rate terms, K-face, melt-input, sliding-law)
bundled together with the observer that records output. Build one with the keyword constructor
below (not this positional one directly), then call [`run!`](@ref).
"""
struct Simulation{F <: AbstractFloat, P <: ModelParameters{F}, HS <: AbstractHeadScheme, GS <: AbstractGapScheme, MT <: MeltTerms, OSS <: AbstractOpenBySlidingScheme, CLS <: AbstractCreepLengthScheme, DS <: AbstractDiffusionScheme, O <: AbstractObserver, G <: Grid, S <: State, MI <: AbstractMeltInput, KFS <: AbstractKFaceScheme, SL <: AbstractSlidingLaw, CNC <: AbstractCellNClamping, TS <: AbstractTimeStepScheme, HE <: AbstractHeadExtrapolation, CF <: AbstractCavityFilling, HP <: AbstractHydraulicPotential, UP <: AbstractIceUplift, GCP <: AbstractGapCoupling, WB <: WaterBudget{F}}
    tsteps::Int
    dt::Base.RefValue{F} # a Ref so AdaptiveTimeStep can update it in place each step, same reason total_time is a Ref despite Simulation itself being immutable
    p::P
    hs::HS
    gs::GS
    mt::MT
    oss::OSS
    cls::CLS # AbstractCreepLengthScheme; StandardCreep() unless p.b_c != 0 (see gap_height.jl)
    ds::DS # AbstractDiffusionScheme; NoDiffusion() unless explicitly passed (see linear_solver.jl) -- not auto-inferred from any ModelParameters field, since WithDiffusion bundles its own solver
    observer::O
    grid::G
    state::S
    mi::MI
    kfs::KFS
    sl::SL
    verbose::Bool
    total_time::Base.RefValue{F} # elapsed simulation time in seconds; a Ref so run!/step_h! can update it in place despite Simulation itself being immutable (same reason PicardSolver -- nested under hs -- is a mutable struct)
    cnc::CNC # per-cell N-clamping override, see cell_N_clamping.jl; NoCellNClamping() (a no-op) by default
    ts::TS # AbstractTimeStepScheme; FixedTimeStep() (a no-op, dt never changes) by default
    he::HE # AbstractHeadExtrapolation: the head solve's initial guess, see head_extrapolation.jl
    cf::CF # AbstractCavityFilling: UnfilledCavities() by default (FilledCavities(): SHAKTI's always water-filled gap), see unfilled_cavities.jl
    hp::HP # AbstractHydraulicPotential: SheetTopPotential() (q = -K grad(h + b)) by default, BedPotential() for the legacy flux, see hydraulic_potential.jl
    up::UP # AbstractIceUplift: IceUplift() by default (NoUplift(): SHAKTI's, pw may exceed po), see ice_uplift.jl
    gcp::GCP # AbstractGapCoupling: ConsistentGapCoupling() by default where available, see the constructor and linear_solver.jl
    wb::WB # WaterBudget: the water each step created beyond its sources and fluxes (water_created), see unfilled_cavities.jl
end

"""
$(TYPEDSIGNATURES)

Builds a [`Simulation`](@ref): `tsteps` steps of size `dt` (seconds), on `grid`/`state`, with
model parameters `p`, melt input `mi`, and sliding law `sl`.

# Notes

- Head scheme: [`PicardHeadScheme`](@ref) (requires `ps`, a [`PicardSolver`](@ref)); with `p.e_v > 0` its rows store
  englacial water (the parabolic head equation).
- `gap_scheme`: [`FullyImplicitGapScheme`](@ref) (default), [`ExplicitGapScheme`](@ref) (a reference, stable only at
  short `dt`), or [`ImplicitGapScheme`](@ref) (deprecated); see [`AbstractGapScheme`](@ref). Ignored under
  [`WithDiffusion`](@ref), whose gap solve has a single scheme of its own (`solve_b_diffusion!`).
- Melt-rate terms: [`MeltTerms`](@ref)'s five flags are read directly off `p.mdot_includes_G`/
  `p.mdot_includes_frictional`/`p.mdot_includes_potential`/`p.mdot_includes_sensible`/
  `p.mdot_includes_qT` (each defaults to `true` in [`ModelParameters`](@ref)).
- Opening-by-sliding scheme: off automatically ([`NoOpenBySliding`](@ref)) if `p.br` is zero, else
  [`WithOpenBySliding`](@ref).
- Creep-length scheme: [`StandardCreep`](@ref) (`l_c = b`) automatically if `p.b_c` is zero, else
  [`CreepCutoff`](@ref) (see `compute_b_fully_implicit_kernel!` for how this combines with
  [`FullyImplicitGapScheme`](@ref)).
- `diffusion_scheme`: [`NoDiffusion`](@ref) (default) or [`WithDiffusion`](@ref) (`linear_solver.jl`)
  -- unlike the opening-by-sliding/creep-length schemes above, this is never auto-inferred from a
  `ModelParameters` field (there's no single physical parameter that makes the diffusion term
  identically zero the way `p.br`/`p.b_c` do): pass `WithDiffusion(ls)` explicitly, with `ls` a
  second, independent [`AbstractLinearSolver`](@ref) instance for `b`'s own diffusion operator.
- `sheet`: [`GlaDS2Sheet`](@ref) (default) or [`ShaktiSheet`](@ref), a [`Sheet`](@ref) bundling `cavity_filling`, `uplift`,
  `hydraulic_potential` and `gap_coupling`. Each of these keywords, when passed, overrides the sheet's field.
- `k_face_choice`: `"arithmetic"`, `"harmonic"`, `"upwind"` or `"implicit_upwind"`, or the scheme itself (`Arithmetic()`,
  `ImplicitUpwind()`, ...; see [`AbstractKFaceScheme`](@ref));
  by default `"implicit_upwind"` under [`UnfilledCavities`](@ref) or [`IceUplift`](@ref) (so with the default sheet),
  `"arithmetic"` otherwise. Where the water depth is free (a free surface or an uplift layer) the flux is a kinematic
  wave in it, `q ~ K(W)*grad(zb)`; with `W` lagged to the start of the step a face mean is unstable (explicit central
  advection) at any `dt`, and `"upwind"` is stable only for `dt < dx*W/(3|q|)` (minutes at 100 m).
  `"implicit_upwind"` ([`ImplicitUpwind`](@ref)) takes `W` at the end of the step and is stable at any `dt`; it needs
  a Picard solver with [`LUDirectSolver`](@ref) or `CUDSSDirectSolver(grid)` (`structure = :general`, the default) in `ps`, and
  `Simulation` errors otherwise. `"upwind"` is the faster opt-out at short steps.
- `cell_N_clamping`: [`NoCellNClamping`](@ref) (default, no-op) or a [`CellNClamping`](@ref)
  overriding specific cells' `N_min`/`N_max` on top of `p`'s global values (see
  `cell_N_clamping.jl`).
- `cavity_filling`: [`UnfilledCavities`](@ref) (default: the gap may be partly empty, with `p_w = 0` there; see
  `unfilled_cavities.jl`) or [`FilledCavities`](@ref) (SHAKTI's original sheet, Sommers et al. 2018: the gap is always
  full of water, and `p_w` can fall below 0 where sliding opens the gap faster than water arrives).
- `hydraulic_potential`: [`SheetTopPotential`](@ref) (default: the potential includes the thickness of the
  flowing water, GlaDS-2 Eq. 7: `q = -K grad(h + b)`; see `hydraulic_potential.jl`) or [`BedPotential`](@ref)
  (legacy, Sommers et al. 2018: the flux is driven by the bed-referenced head, `q = -K grad(h)`). The water's
  level is `h + b` in every regime (`h` is the head at the bed in a filled cell, `z_b + W - b` with a free surface,
  `h_o + W - b` with an uplift layer), so `grad(h)` misses `-grad(b)`: water (and dissipation heat) would not follow
  differences in gap size, or would move under a flat water surface. Pass `BedPotential()` to reproduce the
  original equations bit for bit.
- `uplift`: [`IceUplift`](@ref) (default: the pressure is capped at the overburden and the extra water lifts the
  ice, stored in an uplift layer at `N = 0`; see `ice_uplift.jl`) or [`NoUplift`](@ref) (SHAKTI's original: the water
  pressure may exceed the overburden, `N < 0`). The defaults, `UnfilledCavities()` and `IceUplift()` with
  `hydraulic_potential = SheetTopPotential()`, are the sheet of GlaDS-2 (Wells et al. 2026), with the pressure bounded
  by `0 <= p_w <= p_o` (the range a coupled ice model's sliding law expects). Pass `sheet = ShaktiSheet()` for SHAKTI's
  sheet (`ShaktiSheet(original = true)` for its original equations bit for bit).
- Observer: [`NoObserver`](@ref) if `tracked_obs` is empty; otherwise `which_observer` must be
  `"IO"` (writes to `path` via `which_file_writer`, one of `"NetCDF"`/`"HDF5"`/`"JLD2"`/`"CSV"`,
  at `tracked_times`) or `"Live"` (keeps `tracked_times` in memory instead of writing to disk).
  Or pass the observer itself as `observer` (with `tracked_obs` empty), e.g.
  `observer = IOObserver(["h", "b"], times, NetCDFFileWriter(), path)`.
- `gap_coupling`: [`ExplicitGapCoupling`](@ref) (legacy: the head rows assume the gap changes by `dt*rate(N, b_old)`) or
  [`ConsistentGapCoupling`](@ref) (they assume the change the gap scheme itself makes, so the head equation and the gap update
  store the same water at any `dt`; needed with the implicit and fully implicit gap schemes when `dt*A*|N|^(n-1)*N` is not
  small, and with [`UnfilledCavities`](@ref)/[`IceUplift`](@ref) at large `dt`). Default: `ConsistentGapCoupling()`. Pass `ExplicitGapCoupling()` to reproduce results from before this default. With
  `"implicit_upwind"` the conductance then also takes the end-of-step gap, so the head-gap step is implicit throughout.
  With [`WithDiffusion`](@ref) (SUHMO) `ConsistentGapCoupling()` solves the coupled gap system at every Picard iteration
  (`DiffusiveGapUpdate`), one extra linear solve per iteration (two with `"implicit_upwind"`), with any cavity
  filling and uplift.
- `timestep_scheme`: [`FixedTimeStep`](@ref) (default, `dt` never changes) or
  [`AdaptiveTimeStep`](@ref) (`dt` chosen every step by how the head solve converges, a step that does not converge
  retried at half the `dt`). With the latter, `dt` is the step the controller starts from and `tsteps` should be sized from
  `AdaptiveTimeStep`'s own `dt_min`/`target_time` (see its docstring) rather than from `dt` itself,
  since the true step count isn't known in advance.
- `head_extrapolation_order`: degree of the time extrapolation of the head used as each step's
  initial guess ([`HeadExtrapolation`](@ref); default `1`, linear). `0` starts every solve from the
  previous step's head.
"""
function Simulation(grid, state, tsteps, dt, p, tracked_obs::Vector{String}, mi::AbstractMeltInput, sl::AbstractSlidingLaw; gap_scheme::AbstractGapScheme = FullyImplicitGapScheme(), ps = nothing, which_observer = nothing, which_file_writer = nothing, tracked_times = nothing, path = nothing, k_face_choice = nothing, verbose = false, cell_N_clamping::AbstractCellNClamping = NoCellNClamping(), timestep_scheme::AbstractTimeStepScheme = FixedTimeStep(), diffusion_scheme::AbstractDiffusionScheme = NoDiffusion(), head_extrapolation_order::Int = 1, sheet::Sheet = GlaDS2Sheet(), cavity_filling::Union{Nothing, AbstractCavityFilling} = nothing, hydraulic_potential::Union{Nothing, AbstractHydraulicPotential} = nothing, uplift::Union{Nothing, AbstractIceUplift} = nothing, gap_coupling::Union{Nothing, AbstractGapCoupling} = nothing, observer::Union{Nothing, AbstractObserver} = nothing)

    # The sheet, field by field: a keyword overrides the preset.
    cavity_filling = something(cavity_filling, sheet.cf)
    uplift = something(uplift, sheet.up)
    hydraulic_potential = something(hydraulic_potential, sheet.hp)
    gap_coupling = something(gap_coupling, sheet.gcp)

    # An observer passed as an object names its own observables.
    if observer !== nothing
        isempty(tracked_obs) && which_observer === nothing ||
            error("pass either observer (an object) or tracked_obs with which_observer, not both")
        tracked_obs = observer isa NoObserver ? String[] : observer.tracked_obs
    end

    # Check that all tracked observables are valid State fields
    for name in tracked_obs
        hasfield(typeof(state), Symbol(name)) || name in DERIVED_OBSERVABLES || error("Unknown tracked observable: \"$name\" is not a field of State")
    end

    # Head scheme setup
    ps isa PicardSolver || error("ps (a PicardSolver) must be provided")
    p.e_v >= 0 || error("e_v must be >= 0 (got $(p.e_v))")
    hs = PicardHeadScheme(ps)

    gs = gap_scheme
    gs isa ImplicitGapScheme &&
        @warn "ImplicitGapScheme is deprecated and will be removed: FullyImplicitGapScheme (the default) costs the same and is stable at any dt (ImplicitGapScheme's lagged opening is not)" maxlog = 1

    # Melt-rate terms setup: each flag decided directly by its own ModelParameters field (see
    # MeltTerms, melt_rate.jl) -- no longer inferred from ct/cw being zero.
    mt = MeltTerms{p.mdot_includes_G, p.mdot_includes_frictional, p.mdot_includes_potential, p.mdot_includes_sensible, p.mdot_includes_qT}()

    # Opening-by-sliding scheme setup: off automatically if p.br (the sole
    # factor in compute_beta!'s term, see compute_beta_kernel!) is zero.
    oss = iszero(p.br) ? NoOpenBySliding() : WithOpenBySliding()

    # Creep-length scheme setup: CreepCutoff automatically if p.b_c (the sole
    # factor in creep_length's cutoff branch, see gap_height.jl) is nonzero.
    cls = iszero(p.b_c) ? StandardCreep() : CreepCutoff()

    # K-face averaging scheme setup. Unfilled cavities need the donor-cell (upwind) face conductance, see Upwind
    # (k_face_scheme.jl): with an average, a nearly empty cell discharges through its neighbour's much larger conductance.
    # Under a free surface the implicit water depth is the default; averaged faces are refused there (below).
    free_surface = cavity_filling isa UnfilledCavities || uplift isa IceUplift
    k_face_default = k_face_choice === nothing
    kfs = k_face_scheme(k_face_default ? (free_surface ? ImplicitUpwind() : Arithmetic()) : k_face_choice)
    k_face_choice = k_face_name(kfs)
    # An averaged face lets an unfilled cell's outflow outgrow its water: the arithmetic mean gives a dry cell half its
    # neighbour's conductance, and the water it then delivers is created by the W >= 0 floor of update_water_budget!
    # (Thwaites 2 km, uplift, dt = 1 h: 1.7e7 m^3/s after 12 h against 500 m^3/s of melt, lifting the ice by hundreds of
    # metres). The harmonic mean closes every face of a dry cell, so it never wets again. Both are refused under a free
    # surface. "upwind" lags the water depth: correct, but stable only for dt < dx*W/(3|q|), minutes at 100 m (at 3 h a
    # non-finite gap in step 1 without uplift, 500 Picard iterations and b = 1.7 m with it); it is kept, with a warning,
    # as the only free-surface face for the symmetric solvers (Cholesky, CG, matrix-free).
    if free_surface && (kfs isa Arithmetic || kfs isa Harmonic)
        error("k_face_choice = \"$k_face_choice\" with $(nameof(typeof(cavity_filling)))/$(nameof(typeof(uplift))) does not conserve water: " *
              (kfs isa Arithmetic ? "a dry cell discharges through half its neighbour's conductance and the water it lacks is created. " :
                                    "a dry cell's faces have no conductance, so it never refills. ") *
              "Use \"implicit_upwind\" (the default), or cavity_filling = FilledCavities(), uplift = NoUplift() for SHAKTI's original filled sheet.")
    end
    free_surface && kfs isa Upwind && !k_face_default &&
        @warn "k_face_choice = \"upwind\" with $(nameof(typeof(cavity_filling)))/$(nameof(typeof(uplift))) lags the water depth: stable only for " *
              "dt < dx*W/(3|q|) (minutes at 100 m), above which the gap overshoots or the Picard iteration cycles. Use \"implicit_upwind\" (the default) at longer dt."
    if kfs isa ImplicitUpwind
        # The Newton terms of the donors' water depths make the head matrix non-symmetric (add_water_depth_newton!).
        solver = hs.ps
        hasproperty(solver, :ls) && supports_nonsymmetric(solver.ls) ||
            error("k_face_choice = \"implicit_upwind\"" * (k_face_default ? " (the default with $(nameof(typeof(cavity_filling)))/$(nameof(typeof(uplift))))" : "") *
                  " assembles a non-symmetric head matrix: it needs a PicardSolver " *
                  "with LUDirectSolver(grid), CUDSSDirectSolver(grid) (structure = :general) or, for fields on the GPU, BiCGStabIterativeSolver(grid) as its linear solver, got $(nameof(typeof(solver)))" *
                  (hasproperty(solver, :ls) ? " with $(nameof(typeof(solver.ls)))" : "") *
                  (k_face_default ? ". Alternatively pass k_face_choice = \"upwind\" (stable only at short dt), or cavity_filling = FilledCavities(), " *
                                    "uplift = NoUplift() for SHAKTI's original filled sheet." : ""))
    end

    # A gap solver without a method for the gap system: refused here rather than at the first step (a MethodError deep
    # inside the Picard loop).
    diffusion_scheme isa WithDiffusion &&
        !hasmethod(solve_b_diffusion!, Tuple{typeof(diffusion_scheme.ls), State, Grid, ModelParameters, Any}) &&
        error("WithDiffusion's gap solve has no method for $(nameof(typeof(diffusion_scheme.ls))): use CholeskyDirectSolver(grid), CUDSSDirectSolver(grid; structure = :spd) or CGIterativeSolver (the gap system is symmetric positive definite)")

    # Observer setup
    if observer !== nothing
    elseif isempty(tracked_obs)
        observer = NoObserver()
    elseif which_observer === nothing
        error("which_observer should be specified as \"IO\" or \"Live\" when tracked_obs is not empty.")
    elseif which_observer == "IO"
        if which_file_writer === nothing
            error("which_file_writer should be specified as \"NetCDF\", \"HDF5\", \"JLD2\", or \"CSV\" when which_observer is \"IO\".")
        elseif which_file_writer == "NetCDF"
            fr = NetCDFFileWriter()
        elseif which_file_writer == "HDF5"
            fr = HDF5FileWriter()
        elseif which_file_writer == "JLD2"
            fr = JLD2FileWriter()
        elseif which_file_writer == "CSV"
            fr = CSVFileWriter()
        else
            error("Unknown which_file_writer: \"$which_file_writer\" (expected \"NetCDF\", \"HDF5\", \"JLD2\", or \"CSV\")")
        end
        if tracked_times === nothing
            error("tracked_times should be specified when which_observer is \"IO\".")
        end
        if path === nothing
            error("path should be specified when which_observer is \"IO\".")
        end
        observer = IOObserver(tracked_obs, tracked_times, fr, path)
    elseif which_observer == "Live"
        if tracked_times === nothing
            error("tracked_times should be specified when which_observer is \"Live\".")
        end
        observer = LiveObserver(tracked_obs, tracked_times)
    else
        error("Unknown which_observer: \"$which_observer\" (expected \"IO\" or \"Live\")")
    end

    head_extrapolation_order >= 0 || error("head_extrapolation_order must be >= 0 (got $head_extrapolation_order)")
    he = head_extrapolation_order == 0 ? NoHeadExtrapolation() : HeadExtrapolation(grid; order = head_extrapolation_order)

    # Gap coupling: by default the consistent one wherever it is available. The explicit coupling's head rows assume linear
    # gap growth while the implicit gap schemes apply their own update, so the two store different water: up to 1.8 times
    # the melt input of a 1-day step under WithDiffusion (a moulin slab; Helheim at 500 m: 17 % of the melt after 5 days,
    # against 0.1 % consistent), 22190 m^3/s on a GrIS cold start, and at N < 0 (a filled moulin slab, NoDiffusion, 1-day
    # steps) a gap of 1e269 m in one step, where the consistent coupling keeps it at metres (check_gap_height, run.jl); on
    # Helheim at 500 m (filled, NoDiffusion) the explicit gap reaches 1e10 m with 5-day steps, the consistent one 7 m, for
    # 1 to 2 times the Picard iterations and about the same time per step at 1-day steps.
    # N clamps tighter than [0, Inf) contradict the free-surface rows: the uplifted row assumes N = 0 and the unfilled row
    # N = po while the gap closes at the clamped N, the difference stays in the water W (or is created by its floor W >= 0), and the N handed to
    # an ice model is no longer po - pw. Under those sheets N is in [0, po] (IceUplift and UnfilledCavities) by construction.
    if free_surface
        lo, hi = N_clamp_range(cell_N_clamping)
        (max(lo, p.N_min) > 0 || min(hi, p.N_max) < Inf) &&
            error("N_min > 0, N_max < Inf or a CellNClamping bound tighter than [0, Inf) with $(nameof(typeof(cavity_filling)))/$(nameof(typeof(uplift))) " *
                  "contradicts the unfilled (N = po) and uplifted (N = 0) head rows. These sheets bound N by po - pw already; for a floor " *
                  "on N use IceUplift (water above the overburden lifts the ice), or use the filled sheet (FilledCavities(), NoUplift()).")
    end

    sim = Simulation(tsteps, Ref(dt), p, hs, gs, mt, oss, cls, diffusion_scheme, observer, grid, state, mi, kfs, sl, verbose, Ref(zero(dt)), cell_N_clamping, timestep_scheme, he, cavity_filling, hydraulic_potential, uplift, gap_coupling, WaterBudget(state))
    return refresh_state!(sim) # the derived fields with this simulation's options (set_initial_conditions! knows none of them)

end

"""
$(TYPEDSIGNATURES)

The same [`Simulation`](@ref), with the gap scheme named by a string in the position before `tracked_obs`:
`"fully_implicit"` ([`FullyImplicitGapScheme`](@ref)), `"explicit"` ([`ExplicitGapScheme`](@ref)) or `"implicit"`
([`ImplicitGapScheme`](@ref), deprecated). Kept so that existing drivers run unchanged; the keyword `gap_scheme`
is the preferred form.
"""
function Simulation(grid, state, tsteps, dt, p, gap_scheme_choice::AbstractString, tracked_obs::Vector{String}, mi::AbstractMeltInput, sl::AbstractSlidingLaw; kwargs...)
    gs = gap_scheme_choice == "explicit"       ? ExplicitGapScheme() :
         gap_scheme_choice == "implicit"       ? ImplicitGapScheme() :
         gap_scheme_choice == "fully_implicit" ? FullyImplicitGapScheme() :
         error("Unknown gap_scheme_choice: \"$gap_scheme_choice\" (expected \"explicit\", \"implicit\", or \"fully_implicit\")")
    return Simulation(grid, state, tsteps, dt, p, tracked_obs, mi, sl; gap_scheme = gs, kwargs...)
end

"""
$(TYPEDSIGNATURES)

The same [`Simulation`](@ref) without `tracked_obs`: the output is the `observer` keyword's object
(`IOObserver(obs, times, NetCDFFileWriter(), path)`, `LiveObserver(obs, times)`; [`NoObserver`](@ref) by default),
and the model is the `sheet` keyword's ([`GlaDS2Sheet`](@ref) by default, or [`ShaktiSheet`](@ref)). The preferred form.
"""
Simulation(grid, state, tsteps, dt, p, mi::AbstractMeltInput, sl::AbstractSlidingLaw; kwargs...) =
    Simulation(grid, state, tsteps, dt, p, String[], mi, sl; kwargs...)
