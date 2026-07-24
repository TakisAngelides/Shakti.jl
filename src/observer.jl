abstract type AbstractFileWriter end

struct NetCDFFileWriter <: AbstractFileWriter end
struct HDF5FileWriter <: AbstractFileWriter end
struct JLD2FileWriter <: AbstractFileWriter end
struct CSVFileWriter <: AbstractFileWriter end

abstract type AbstractObserver end

struct NoObserver <: AbstractObserver end

struct IOObserver{FR <: AbstractFileWriter} <: AbstractObserver
    tracked_obs::Vector{String}   # names of State fields to record
    tracked_times::AbstractVector{Int}      # time step indices at which to record
    fr::FR                        # file format
    path::String                  # where to write
    handle::Ref{Any}              # set by prepare!; holds the open file/dataset handle
end

IOObserver(tracked_obs, tracked_times, fr, path) = IOObserver(tracked_obs, tracked_times, fr, path, Ref{Any}(nothing))

struct LiveObserver <: AbstractObserver # no writing to files, just arrays kept in RAM
    tracked_obs::Vector{String}   # names of State fields to record
    tracked_times::AbstractVector{Int}      # time step indices at which to record
    history::Dict{String, Array}  # set by prepare!; one preallocated array per tracked observable
end

LiveObserver(tracked_obs, tracked_times) = LiveObserver(tracked_obs, tracked_times, Dict{String, Array}())

# Resolves a tracked observable's name (as given by the user in tracked_obs)
# to the actual State field, e.g. get_observable(state, "h") -> state.h.
# Validity of `name` (i.e. that it's really a State field) is checked once,
# at Simulation-construction time (see simulation.jl), not on every call here.
get_observable(state::State, name::String) = getfield(state, Symbol(name))

# =============================================================================
# Observer setup, called once before the time loop
# =============================================================================

prepare!(observer::NoObserver, state::State) = nothing

function prepare!(observer::LiveObserver, state::State)
    for name in observer.tracked_obs
        field = get_observable(state, name)
        observer.history[name] = Array{eltype(field)}(undef, size(field)..., length(observer.tracked_times))
    end
    return nothing
end

function prepare!(observer::IOObserver, state::State)
    observer.handle[] = openfile!(observer.fr, observer, state)
    return nothing
end

# NetCDF/HDF5/JLD2 record the full tracked fields at each tracked time (like
# LiveObserver, just written to disk instead of kept in RAM): one dataset per
# tracked field, sized (field's own shape..., length(tracked_times)), written
# one time-slice at a time. CSV instead records one row per tracked time, with
# min/max/mean columns per tracked field -- unlike the others it's meant for
# scalar summaries (e.g. domain-averaged melt rate over time), not full grids.
csv_stat_colnames(tracked_obs) = (:t, :total_time, (Symbol(name * suffix) for name in tracked_obs for suffix in ("_min", "_max", "_mean"))...)

function openfile!(fr::NetCDFFileWriter, observer::IOObserver, state::State)
    ntimes = length(observer.tracked_times)
    tdim = NcDim("time", ntimes)
    vars = NcVar[NcVar("time", [tdim]; t = Float64)]
    for name in observer.tracked_obs
        field = get_observable(state, name)
        xdim = NcDim("$(name)_x", size(field, 1))
        ydim = NcDim("$(name)_y", size(field, 2))
        push!(vars, NcVar(name, [xdim, ydim, tdim]; t = eltype(field)))
    end
    return NetCDF.create(observer.path, vars)
end

function openfile!(fr::HDF5FileWriter, observer::IOObserver, state::State)
    file = HDF5.h5open(observer.path, "w")
    ntimes = length(observer.tracked_times)
    HDF5.create_dataset(file, "time", Float64, (ntimes,))
    for name in observer.tracked_obs
        field = get_observable(state, name)
        HDF5.create_dataset(file, name, eltype(field), (size(field)..., ntimes))
    end
    return file
end

function openfile!(fr::JLD2FileWriter, observer::IOObserver, state::State)
    file = JLD2.jldopen(observer.path, "w")
    file["tracked_obs"] = observer.tracked_obs
    file["tracked_times"] = observer.tracked_times
    return file
end

# No persistent handle: each write2file! call below opens/appends/closes on
# its own (CSV.write has no notion of a long-lived writable handle), so this
# only needs to lay down the header row up front.
function openfile!(fr::CSVFileWriter, observer::IOObserver, state::State)
    colnames = csv_stat_colnames(observer.tracked_obs)
    coltypes = Tuple{eltype(observer.tracked_times), Float64, ntuple(_ -> Float64, 3 * length(observer.tracked_obs))...}
    CSV.write(observer.path, NamedTuple{colnames, coltypes}[])
    return observer.path
end

# =============================================================================
# Resume setup, called instead of prepare! when restarting from a checkpoint
# (see checkpoint.jl/run.jl): reopens an IOObserver's existing output file
# for further writes instead of truncating it via openfile!, so the resumed
# run continues writing into the same file. NoObserver has nothing to resume;
# LiveObserver's history lived only in the crashed process's RAM and can't be
# recovered, so it just starts a fresh (gap-containing) buffer via prepare!.
# =============================================================================

resume!(observer::NoObserver, state::State, resume_t::Int) = nothing
resume!(observer::LiveObserver, state::State, resume_t::Int) = prepare!(observer, state)

function resume!(observer::IOObserver, state::State, resume_t::Int)
    observer.handle[] = reopenfile!(observer.fr, observer, state, resume_t)
    return nothing
end

# NetCDF/HDF5 datasets are preallocated to their full (fields..., ntimes)
# shape up front, so reopening for write and overwriting a given time index
# is always well-defined -- no special handling needed for a tstep the
# crashed run already wrote.
reopenfile!(fr::NetCDFFileWriter, observer::IOObserver, state::State, resume_t::Int) = NetCDF.open(observer.path; mode = NetCDF.NC_WRITE)
reopenfile!(fr::HDF5FileWriter, observer::IOObserver, state::State, resume_t::Int) = HDF5.h5open(observer.path, "r+")

# JLD2 has no preallocated shape (see write2file! above); reopening for write
# is enough since write2file! itself is overwrite-safe.
reopenfile!(fr::JLD2FileWriter, observer::IOObserver, state::State, resume_t::Int) = JLD2.jldopen(observer.path, "r+")

# CSV has no persistent handle or indexed slots -- each write2file! call
# appends a new row. A row for some t in (resume_t, crash_t] may already be
# in the file from the crashed run (it wrote up through crash_t, but the
# checkpoint being resumed from is only current up to resume_t <= crash_t);
# replaying tsteps resume_t+1:crash_t would otherwise duplicate those rows,
# so trim them here before resuming appends.
function reopenfile!(fr::CSVFileWriter, observer::IOObserver, state::State, resume_t::Int)
    rows = CSV.File(observer.path)
    keep = filter(row -> row.t <= resume_t, rows)
    CSV.write(observer.path, keep)
    return observer.path
end

# =============================================================================
# Per-time-step observation
# =============================================================================

observe!(observer::NoObserver, state::State, t, total_time) = nothing

function observe!(observer::LiveObserver, state::State, t, total_time)
    idx = findfirst(==(t), observer.tracked_times)
    idx === nothing && return nothing
    for name in observer.tracked_obs
        hist = observer.history[name]
        selectdim(hist, ndims(hist), idx) .= Array(get_observable(state, name))
    end
    return nothing
end

function observe!(observer::IOObserver, state::State, t, total_time)
    idx = findfirst(==(t), observer.tracked_times)
    idx === nothing && return nothing
    write2file!(observer.fr, observer, state, idx, total_time)
    return nothing
end

function write2file!(fr::NetCDFFileWriter, observer::IOObserver, state::State, idx::Int, total_time::AbstractFloat)
    nc = observer.handle[]
    NetCDF.putvar(nc, "time", [total_time]; start = [idx], count = [1])
    for name in observer.tracked_obs
        field = Array(get_observable(state, name))
        NetCDF.putvar(nc, name, field; start = [1, 1, idx], count = [size(field, 1), size(field, 2), 1])
    end
    return nothing
end

function write2file!(fr::HDF5FileWriter, observer::IOObserver, state::State, idx::Int, total_time::AbstractFloat)
    file = observer.handle[]
    file["time"][idx] = total_time
    for name in observer.tracked_obs
        file[name][:, :, idx] = Array(get_observable(state, name))
    end
    return nothing
end

function write2file!(fr::JLD2FileWriter, observer::IOObserver, state::State, idx::Int, total_time::AbstractFloat)
    file = observer.handle[]
    jld2_set!(file, "total_time/$idx", total_time)
    for name in observer.tracked_obs
        jld2_set!(file, "$name/$idx", Array(get_observable(state, name)))
    end
    return nothing
end

# JLD2 errors on reassigning an existing key, unlike NetCDF/HDF5's indexed
# writes which just overwrite in place -- this makes JLD2 writes overwrite-safe
# too, so a resumed run redoing a tstep whose output the crashed run already
# wrote (see resume!/reopenfile! below) doesn't error.
function jld2_set!(file, key::String, value)
    haskey(file, key) && delete!(file, key)
    file[key] = value
    return nothing
end

function write2file!(fr::CSVFileWriter, observer::IOObserver, state::State, idx::Int, total_time::AbstractFloat)
    colnames = csv_stat_colnames(observer.tracked_obs)
    stats = Float64[]
    for name in observer.tracked_obs
        field = get_observable(state, name)
        mn, mx = extrema(field)
        push!(stats, mn, mx, mean(field))
    end
    row = NamedTuple{colnames}((observer.tracked_times[idx], total_time, stats...))
    CSV.write(observer.path, [row]; append = true)
    return nothing
end

# =============================================================================
# Teardown, called once after the time loop
# =============================================================================

finalize!(observer::AbstractObserver, state::State) = nothing # NoObserver/LiveObserver hold nothing to close

function finalize!(observer::IOObserver, state::State)
    close_handle!(observer.fr, observer.handle[])
    return nothing
end

close_handle!(fr::NetCDFFileWriter, handle) = nothing # NetCDF.jl closes NcFile handles via finalizer, not an explicit close
close_handle!(fr::HDF5FileWriter, handle) = HDF5.close(handle)
close_handle!(fr::JLD2FileWriter, handle) = JLD2.close(handle)
close_handle!(fr::CSVFileWriter, handle) = nothing # each write2file! call already opens/appends/closes on its own
