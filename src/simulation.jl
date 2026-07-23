abstract type AbstractHeadScheme end

struct ParabolicHeadScheme <: AbstractHeadScheme end
struct EllipticHeadScheme{PS <: PicardSolver} <: AbstractHeadScheme
    ps::PS
end

abstract type AbstractGapScheme end

struct ExplicitGapScheme <: AbstractGapScheme end
struct ImplicitGapScheme <: AbstractGapScheme end

struct Simulation{F <: AbstractFloat, P <: ModelParameters{F}, HS <: AbstractHeadScheme, GS <: AbstractGapScheme, SHS <: AbstractSensibleHeatScheme, O <: AbstractObserver, G <: Grid, S <: State, MI <: AbstractMeltInput, KFS <: AbstractKFaceScheme, SL <: AbstractSlidingLaw}
    tsteps::Int
    dt::F
    p::P
    hs::HS
    gs::GS
    shs::SHS
    observer::O
    grid::G
    state::S
    mi::MI
    kfs::KFS
    sl::SL
    verbose::Bool
    total_time::Base.RefValue{F} # elapsed simulation time in seconds; a Ref so run!/step_h! can update it in place despite Simulation itself being immutable (same reason PicardSolver -- nested under hs -- is a mutable struct)
end

function Simulation(grid, state, tsteps, dt, p, gap_scheme_choice, tracked_obs::Vector{String}, mi::AbstractMeltInput, sl::AbstractSlidingLaw; ps = nothing, which_observer = nothing, which_file_writer = nothing, tracked_times = nothing, path = nothing, k_face_choice = "arithmetic", verbose = false)

    # Check that all tracked observables are valid State fields
    for name in tracked_obs
        hasfield(typeof(state), Symbol(name)) || error("Unknown tracked observable: \"$name\" is not a field of State")
    end

    # Head scheme setup
    if iszero(p.e_v) # elliptic head scheme
        ps === nothing && error("ps (a PicardSolver) must be provided when p.e_v == 0 (elliptic head scheme)")
        hs = EllipticHeadScheme(ps)
    else # parabolic head scheme
        hs = ParabolicHeadScheme()
    end

    # Gap scheme setup
    if gap_scheme_choice == "explicit"
        gs = ExplicitGapScheme()
    elseif gap_scheme_choice == "implicit"
        gs = ImplicitGapScheme()
    else
        error("Unknown gap_scheme_choice: \"$gap_scheme_choice\" (expected \"explicit\" or \"implicit\")")
    end

    # Sensible-heat scheme setup: off automatically if either factor in its
    # ct*cw prefactor (see compute_mdot!) is zero.
    shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()

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

    return Simulation(tsteps, dt, p, hs, gs, shs, observer, grid, state, mi, kfs, sl, verbose, Ref(zero(dt)))

end
