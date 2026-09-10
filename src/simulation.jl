"""
$(TYPEDSIGNATURES)

How the hydraulic head is advanced each timestep -- multiple dispatch on the concrete subtype
(see [`step_h!`](@ref) in `run.jl`) picks between [`EllipticHeadScheme`](@ref) (Picard iteration,
used when `p.e_v == 0`) and [`ParabolicHeadScheme`](@ref) (single backward-Euler solve, used when
`p.e_v != 0`).
"""
abstract type AbstractHeadScheme end

"""
$(TYPEDSIGNATURES)

Head scheme for `p.e_v != 0` (nonzero englacial storage void ratio, Sommers et al. 2018 Eq. 13's
`∂(e_v(h-zb))/∂t` storage term): each timestep, a single backward-Euler linear solve (`ls`, see
[`parabolic_solver!`](@ref)) updates `h` directly -- no Picard loop. Every nonlinear coefficient
(`K`, `N`, `A_visc`, `b`, `mdot`, `beta`, `abs_ub`) is evaluated at the current (start-of-timestep)
state rather than iterated to convergence, the same lagged-coefficient idiom
`compute_b_implicit_kernel!` uses for `b`.
"""
struct ParabolicHeadScheme{LS <: AbstractLinearSolver} <: AbstractHeadScheme
    ls::LS
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
subtype picks between [`ExplicitGapScheme`](@ref) and [`ImplicitGapScheme`](@ref) (see
`compute_b!` in `gap_height.jl`).
"""
abstract type AbstractGapScheme end

"""
$(TYPEDSIGNATURES)

Explicit (forward-Euler) update of `b`: cheaper per step, but only stable for small enough `dt`.
"""
struct ExplicitGapScheme <: AbstractGapScheme end

"""
$(TYPEDSIGNATURES)

Implicit (backward-Euler) update of `b`: unconditionally stable, the default choice.
"""
struct ImplicitGapScheme <: AbstractGapScheme end

"""
$(TYPEDSIGNATURES)

Everything needed to run a subglacial hydrology simulation: the grid, state, model parameters,
and every "which scheme/law" choice (head, gap, melt-rate terms, K-face, melt-input, sliding-law)
bundled together with the observer that records output. Build one with the keyword constructor
below (not this positional one directly), then call [`run!`](@ref).
"""
struct Simulation{F <: AbstractFloat, P <: ModelParameters{F}, HS <: AbstractHeadScheme, GS <: AbstractGapScheme, MT <: MeltTerms, OSS <: AbstractOpenBySlidingScheme, O <: AbstractObserver, G <: Grid, S <: State, MI <: AbstractMeltInput, KFS <: AbstractKFaceScheme, SL <: AbstractSlidingLaw, CGC <: AbstractCellGapClamping, CNC <: AbstractCellNClamping}
    tsteps::Int
    dt::F
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
end

"""
$(TYPEDSIGNATURES)

Builds a [`Simulation`](@ref): `tsteps` steps of size `dt` (seconds), on `grid`/`state`, with
model parameters `p`, melt input `mi`, and sliding law `sl`.

# Notes

- Head scheme: [`EllipticHeadScheme`](@ref) if `p.e_v == 0` (requires `ps`, a
  [`PicardSolver`](@ref)), else [`ParabolicHeadScheme`](@ref) (requires `ls`, an
  [`AbstractLinearSolver`](@ref)).
- `gap_scheme_choice`: `"explicit"` or `"implicit"` (see [`AbstractGapScheme`](@ref)).
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
"""
function Simulation(grid, state, tsteps, dt, p, gap_scheme_choice, tracked_obs::Vector{String}, mi::AbstractMeltInput, sl::AbstractSlidingLaw; ps = nothing, ls = nothing, which_observer = nothing, which_file_writer = nothing, tracked_times = nothing, path = nothing, k_face_choice = "arithmetic", verbose = false, cell_gap_clamping::AbstractCellGapClamping = NoCellGapClamping(), cell_N_clamping::AbstractCellNClamping = NoCellNClamping())

    # Check that all tracked observables are valid State fields
    for name in tracked_obs
        hasfield(typeof(state), Symbol(name)) || error("Unknown tracked observable: \"$name\" is not a field of State")
    end

    # Head scheme setup
    if iszero(p.e_v) # elliptic head scheme
        ps === nothing && error("ps (a PicardSolver) must be provided when p.e_v == 0 (elliptic head scheme)")
        hs = EllipticHeadScheme(ps)
    else # parabolic head scheme
        ls === nothing && error("ls (an AbstractLinearSolver) must be provided when p.e_v != 0 (parabolic head scheme)")
        hs = ParabolicHeadScheme(ls)
    end

    # Gap scheme setup
    if gap_scheme_choice == "explicit"
        gs = ExplicitGapScheme()
    elseif gap_scheme_choice == "implicit"
        gs = ImplicitGapScheme()
    else
        error("Unknown gap_scheme_choice: \"$gap_scheme_choice\" (expected \"explicit\" or \"implicit\")")
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

    return Simulation(tsteps, dt, p, hs, gs, mt, oss, observer, grid, state, mi, kfs, sl, verbose, Ref(zero(dt)), cell_gap_clamping, cell_N_clamping)

end
