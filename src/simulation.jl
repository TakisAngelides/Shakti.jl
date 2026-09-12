"""
$(TYPEDSIGNATURES)

How the hydraulic head is advanced each timestep -- multiple dispatch on the concrete subtype
(see [`step_h!`](@ref) in `run.jl`) picks between [`EllipticHeadScheme`](@ref) (Picard iteration,
used when `p.e_v == 0`) and [`ParabolicHeadScheme`](@ref) (backward-Euler, iterated to nonlinear
convergence within each timestep, used when `p.e_v != 0`).
"""
abstract type AbstractHeadScheme end

"""
$(TYPEDSIGNATURES)

Head scheme for `p.e_v != 0` (nonzero englacial storage void ratio, Sommers et al. 2018 Eq. 13's
`∂(e_v(h-zb))/∂t` storage term): each timestep, [`Parabolic_loop!`](@ref) (`pps`, a
[`ParabolicPicardSolver`](@ref)) repeats the backward-Euler linear solve (see
[`solve_parabolic_linear_system!`](@ref)), refreshing every nonlinear coefficient (`K`, `N`,
`A_visc`, `b`, `mdot`, `beta`, `abs_ub`) against the newly-updated `h` each time, until the head
update converges (or `pps.iters` is exhausted) -- the parabolic counterpart of
[`EllipticHeadScheme`](@ref)'s Picard loop, needed because a *single* backward-Euler solve leaves
those coefficients lagged at the start-of-timestep state (see [`parabolic_solver!`](@ref)'s
docstring for where that lag causes real, demonstrated error at large `dt`). Pass a
`ParabolicPicardSolver` built with `iters = 1` to recover that old, non-iterating behavior exactly.
"""
struct ParabolicHeadScheme{PPS <: ParabolicPicardSolver} <: AbstractHeadScheme
    pps::PPS
end

"""
$(TYPEDSIGNATURES)

Head scheme used when `p.e_v == 0`: each timestep, `ps` (a [`PicardSolver`](@ref)) is run to
convergence via Picard iteration.
"""
struct EllipticHeadScheme{PS <: PicardSolver} <: AbstractHeadScheme
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
(see [`compute_b_implicit_kernel!`](@ref)'s docstring for the exact bound). Confirmed in practice,
not just in theory: a real AIS 32km run hit this at `dt=900s` and blew `b` up to the `b_max` safety
cap in one cell, something [`FullyImplicitGapScheme`](@ref) -- now the default -- doesn't do.
Unconditionally stable outright when `p.br == 0`.
"""
struct ImplicitGapScheme <: AbstractGapScheme end

"""
$(TYPEDSIGNATURES)

Fully implicit update of `b`: both the creep-closure term AND the opening-by-sliding term are
evaluated at the new (`k+1`) `b`, not the lagged one -- unconditionally stable for any `dt`,
regardless of `p.br`. `beta(b) = max(0, (p.br-b)/p.lr)` is piecewise-linear (not smooth, but not
genuinely nonlinear either), so this doesn't need a Newton iteration: [`compute_b_fully_implicit_kernel!`](@ref)
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
[`AdaptiveTimeStep`](@ref) (recomputed every step from the current `A_visc`/`N`/`u_b` fields).
"""
abstract type AbstractTimeStepScheme end

"""
$(TYPEDSIGNATURES)

`sim.dt[]` is set once at construction and never changes. Today's behavior; the default.
"""
struct FixedTimeStep <: AbstractTimeStepScheme end

"""
$(TYPEDSIGNATURES)

Recomputes `sim.dt[]` every step from the gap-height equation's own local relaxation rate
`C + gamma = A_visc*|N|^(n-1)*N + |u_b|/lr` -- the same quantities behind
[`FullyImplicitGapScheme`](@ref)'s unconditional-stability proof and [`ImplicitGapScheme`](@ref)'s
`dt` cap: `dt = clamp(safety_factor / stat, dt_min, dt_max)`, where `stat` is either the domain
maximum of that rate (`UsePercentile=false`, the default -- a single cheap, GPU-native reduction,
fully conservative) or a chosen percentile of it restricted to grounded cells
(`UsePercentile=true` -- needs a sort each step, and a host-device transfer on a GPU backend, but
avoids one outlier cell dominating). `UsePercentile` is a type parameter, not a field, so the
choice is resolved at compile time into two separate `rate_statistic` methods -- same "dispatch on
a type decided once outside the hot loop" idiom as [`MeltTerms`](@ref) -- not a runtime branch
inside the per-step reduction.

Note this recomputes `dt` from the *previous* step's converged `N`/`u_b` (it can't know the
current step's `N` before solving it) -- the same lagged-coefficient idiom already used for
`s.beta` in [`ImplicitGapScheme`](@ref).

Build with the [`AdaptiveTimeStep(grid, state; ...)`](@ref) convenience constructor below, not
this one directly -- it sizes `rate_field`/`scratch`/`grounded_indices` correctly.
"""
struct AdaptiveTimeStep{UsePercentile, F} <: AbstractTimeStepScheme
    safety_factor::F
    percentile::F                 # quantile of the RATE field C+gamma (not of tau=1/(C+gamma)); unused when UsePercentile=false
    dt_min::F
    dt_max::F
    target_time::F                # physical duration to run for; run! breaks out of its tsteps loop once total_time[] reaches this
    rate_field::AbstractMatrix{F} # preallocated once (same backend as State), holds C+gamma per cell each step
    host_rate::Vector{F}          # preallocated once, CPU-side; empty when UsePercentile=false. Bridges a possibly-GPU-resident rate_field via one bulk copyto! (CUDA.jl/Metal.jl's real device->host transfer -- scalar-indexing a GPU array per grounded cell would be illegal/catastrophically slow, see checkpoint.jl's own note on this)
    scratch::Vector{F}            # preallocated once; empty when UsePercentile=false. Gathered from host_rate via grounded_indices, then sorted in place each step
    grounded_indices::Vector{Int} # linear indices of GROUNDED cells into rate_field, precomputed once from the (static) mask; empty when UsePercentile=false
end

"""
$(TYPEDSIGNATURES)

Builds an [`AdaptiveTimeStep`](@ref), sizing its scratch buffers from `grid`/`state.mask`.

# Arguments
- `use_percentile`: `false` (default) uses the domain maximum of `C+gamma` each step (cheap,
  GPU-native, fully conservative); `true` uses the `percentile`-th quantile of `C+gamma`
  restricted to grounded cells instead (needs a sort; more representative, less swayed by one
  outlier cell).
- `safety_factor`: `dt_candidate = safety_factor / stat` -- e.g. `0.3` for the accuracy-oriented
  guidance worked out empirically for AIS/GrIS/Drang Drung, larger if some inaccuracy is
  acceptable.
- `percentile`: only used when `use_percentile=true`, in RATE space (not `tau` space) -- `0.75`
  means "the value such that 75% of grounded cells have `C+gamma` below it", i.e. the
  *fast-relaxing* end of the distribution, corresponding to the 25th percentile of
  `tau=1/(C+gamma)`.
- `dt_min`/`dt_max`: hard bounds on the resulting `dt`. `dt_min` in particular guarantees
  [`run!`](@ref)'s loop terminates: total simulated time advances by at least `dt_min` every step,
  so a `target_time`-based run takes at most `ceil((target_time-total_time)/dt_min)` steps -- pass
  that same bound as `tsteps` when constructing the [`Simulation`](@ref).
- `target_time`: the physical duration (seconds) this run is meant to cover -- the final step is
  clipped to land exactly on it rather than overshoot.
"""
function AdaptiveTimeStep(grid::Grid, state::State; use_percentile::Bool = false, safety_factor, percentile = 0.75, dt_min, dt_max, target_time)
    F = eltype(state.N)
    rate_field = initialize_center_field(grid)
    if use_percentile
        grounded_indices = findall(vec(Array(state.mask)) .== GROUNDED)
        host_rate = Vector{F}(undef, length(rate_field))
        scratch = Vector{F}(undef, length(grounded_indices))
        return AdaptiveTimeStep{true, F}(F(safety_factor), F(percentile), F(dt_min), F(dt_max), F(target_time), rate_field, host_rate, scratch, grounded_indices)
    else
        return AdaptiveTimeStep{false, F}(F(safety_factor), F(percentile), F(dt_min), F(dt_max), F(target_time), rate_field, F[], F[], Int[])
    end
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

Everything needed to run a subglacial hydrology simulation: the grid, state, model parameters,
and every "which scheme/law" choice (head, gap, melt-rate terms, K-face, melt-input, sliding-law)
bundled together with the observer that records output. Build one with the keyword constructor
below (not this positional one directly), then call [`run!`](@ref).
"""
struct Simulation{F <: AbstractFloat, P <: ModelParameters{F}, HS <: AbstractHeadScheme, GS <: AbstractGapScheme, MT <: MeltTerms, OSS <: AbstractOpenBySlidingScheme, O <: AbstractObserver, G <: Grid, S <: State, MI <: AbstractMeltInput, KFS <: AbstractKFaceScheme, SL <: AbstractSlidingLaw, CGC <: AbstractCellGapClamping, CNC <: AbstractCellNClamping, TS <: AbstractTimeStepScheme}
    tsteps::Int
    dt::Base.RefValue{F} # a Ref so AdaptiveTimeStep can update it in place each step, same reason total_time is a Ref despite Simulation itself being immutable
    p::P
    hs::HS
    gs::GS
    mt::MT
    oss::OSS
    observer::O
    grid::G
    state::S
    mi::MI
    kfs::KFS
    sl::SL
    verbose::Bool
    total_time::Base.RefValue{F} # elapsed simulation time in seconds; a Ref so run!/step_h! can update it in place despite Simulation itself being immutable (same reason PicardSolver -- nested under hs -- is a mutable struct)
    cgc::CGC # per-cell gap-clamping override, see cell_gap_clamping.jl; NoCellGapClamping() (a no-op) by default
    cnc::CNC # per-cell N-clamping override, see cell_N_clamping.jl; NoCellNClamping() (a no-op) by default
    ts::TS # AbstractTimeStepScheme; FixedTimeStep() (a no-op, dt never changes) by default
end

"""
$(TYPEDSIGNATURES)

Builds a [`Simulation`](@ref): `tsteps` steps of size `dt` (seconds), on `grid`/`state`, with
model parameters `p`, melt input `mi`, and sliding law `sl`.

# Notes

- Head scheme: [`EllipticHeadScheme`](@ref) if `p.e_v == 0` (requires `ps`, a
  [`PicardSolver`](@ref)), else [`ParabolicHeadScheme`](@ref) (requires `pps`, a
  [`ParabolicPicardSolver`](@ref)).
- `gap_scheme_choice`: `"explicit"`, `"implicit"`, or `"fully_implicit"` (see [`AbstractGapScheme`](@ref)).
- Melt-rate terms: [`MeltTerms`](@ref)'s five flags are read directly off `p.mdot_includes_G`/
  `p.mdot_includes_frictional`/`p.mdot_includes_potential`/`p.mdot_includes_sensible`/
  `p.mdot_includes_qT` (each defaults to `true` in [`ModelParameters`](@ref)).
- Opening-by-sliding scheme: off automatically ([`NoOpenBySliding`](@ref)) if `p.br` is zero, else
  [`WithOpenBySliding`](@ref).
- `k_face_choice`: `"arithmetic"` or `"harmonic"` (see [`AbstractKFaceScheme`](@ref)).
- `cell_gap_clamping`: [`NoCellGapClamping`](@ref) (default, no-op) or a [`CellGapClamping`](@ref)
  overriding specific cells' `b_min`/`b_max` on top of `p`'s global values (see
  `cell_gap_clamping.jl`).
- `cell_N_clamping`: [`NoCellNClamping`](@ref) (default, no-op) or a [`CellNClamping`](@ref)
  overriding specific cells' `N_min`/`N_max` on top of `p`'s global values (see
  `cell_N_clamping.jl`).
- Observer: [`NoObserver`](@ref) if `tracked_obs` is empty; otherwise `which_observer` must be
  `"IO"` (writes to `path` via `which_file_writer`, one of `"NetCDF"`/`"HDF5"`/`"JLD2"`/`"CSV"`,
  at `tracked_times`) or `"Live"` (keeps `tracked_times` in memory instead of writing to disk).
- `timestep_scheme`: [`FixedTimeStep`](@ref) (default, `dt` never changes) or
  [`AdaptiveTimeStep`](@ref) (`dt` recomputed every step from `C+gamma`). With the latter, `dt` is
  still the *first* step's value (until the first recomputation) and `tsteps` should be sized from
  `AdaptiveTimeStep`'s own `dt_min`/`target_time` (see its docstring) rather than from `dt` itself,
  since the true step count isn't known in advance.
"""
function Simulation(grid, state, tsteps, dt, p, gap_scheme_choice, tracked_obs::Vector{String}, mi::AbstractMeltInput, sl::AbstractSlidingLaw; ps = nothing, pps = nothing, which_observer = nothing, which_file_writer = nothing, tracked_times = nothing, path = nothing, k_face_choice = "arithmetic", verbose = false, cell_gap_clamping::AbstractCellGapClamping = NoCellGapClamping(), cell_N_clamping::AbstractCellNClamping = NoCellNClamping(), timestep_scheme::AbstractTimeStepScheme = FixedTimeStep())

    # Check that all tracked observables are valid State fields
    for name in tracked_obs
        hasfield(typeof(state), Symbol(name)) || error("Unknown tracked observable: \"$name\" is not a field of State")
    end

    # Head scheme setup
    if iszero(p.e_v) # elliptic head scheme
        ps === nothing && error("ps (a PicardSolver) must be provided when p.e_v == 0 (elliptic head scheme)")
        hs = EllipticHeadScheme(ps)
    else # parabolic head scheme
        pps === nothing && error("pps (a ParabolicPicardSolver) must be provided when p.e_v != 0 (parabolic head scheme)")
        hs = ParabolicHeadScheme(pps)
    end

    # Gap scheme setup
    if gap_scheme_choice == "explicit"
        gs = ExplicitGapScheme()
    elseif gap_scheme_choice == "implicit"
        gs = ImplicitGapScheme()
    elseif gap_scheme_choice == "fully_implicit"
        gs = FullyImplicitGapScheme()
    else
        error("Unknown gap_scheme_choice: \"$gap_scheme_choice\" (expected \"explicit\", \"implicit\", or \"fully_implicit\")")
    end

    # Melt-rate terms setup: each flag decided directly by its own ModelParameters field (see
    # MeltTerms, melt_rate.jl) -- no longer inferred from ct/cw being zero.
    mt = MeltTerms{p.mdot_includes_G, p.mdot_includes_frictional, p.mdot_includes_potential, p.mdot_includes_sensible, p.mdot_includes_qT}()

    # Opening-by-sliding scheme setup: off automatically if p.br (the sole
    # factor in compute_beta!'s term, see compute_beta_kernel!) is zero.
    oss = iszero(p.br) ? NoOpenBySliding() : WithOpenBySliding()

    # K-face averaging scheme setup
    if k_face_choice == "arithmetic"
        kfs = Arithmetic()
    elseif k_face_choice == "harmonic"
        kfs = Harmonic()
    else
        error("Unknown k_face_choice: \"$k_face_choice\" (expected \"arithmetic\" or \"harmonic\")")
    end

    # Observer setup
    if isempty(tracked_obs)
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

    return Simulation(tsteps, Ref(dt), p, hs, gs, mt, oss, observer, grid, state, mi, kfs, sl, verbose, Ref(zero(dt)), cell_gap_clamping, cell_N_clamping, timestep_scheme)

end
