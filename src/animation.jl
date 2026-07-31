# =============================================================================
# Animation
# =============================================================================
#
# Turns a LiveObserver's in-RAM history into .mp4 animations. Rendered with
# CairoMakie (software/Cairo rasterizer, via record()'s backend-agnostic
# colorbuffer()) rather than GLMakie -- no OpenGL/display needed, so this
# works on headless HPC nodes as well as locally. LiveObserver.history[name]
# is a single preallocated array of shape (field's own shape...,
# length(tracked_times)) -- time is the last dimension --. Frames are selected 
# here with selectdim(..., ndims(hist), idx).
#
# tracked_times holds the actual simulation time-step index for each saved
# frame, so titles below label frames by real time-step number rather than by
# save order.

"""
$(TYPEDSIGNATURES)

Renders an animated `.mp4` of a 1D slice (row `j`) of a tracked field's time history `hist`
(shape `field's own shape..., ntimes`, e.g. from [`LiveObserver`](@ref)'s `history[name]`) to
`filename`, one frame per tracked time, labeled by the real timestep number (`tracked_times`).
`moulin_ij` (a vector of `(i,j)` grid indices, see [`get_moulin_ij`](@ref)) optionally overlays
moulin locations on row `j` as red markers. Uses CairoMakie's software rasterizer, so this works
on headless HPC nodes as well as locally.
"""
function make_mp4_mid(hist::AbstractArray, tracked_times, j, moulin_ij; filename, show_moulins::Bool = true)
    ntimes = size(hist, ndims(hist)) # number of time steps saved in hist
    ymin   = minimum(hist)
    ymax   = maximum(hist)
    x      = 1:size(hist, 1) # remember in history the data is saved as x, y, t
    # Observable(...) wraps that vector in a Makie Observable so it's the animation's mutable data source — 
    # later frames update data[] in place (does lines!(ax, x, data), which auto-updates when data changes).
    data   = Observable(selectdim(hist, ndims(hist), 1)[:, j]) # from hist we take the configuration of the first index of the last dimension - (which is the time) - and slice spatially [:, j]
    fig    = Figure(size = (1000, 800))
    ax     = Axis(fig[1, 1]; xlabel = "i", ylabel = "Value", limits = (nothing, (ymin, ymax)))
    lines!(ax, x, data)

    moulin_i = show_moulins ? [mi for (mi, mj) in moulin_ij if mj == j] : Int[] # going to put red dots at the x locations where the moulin was during the simulation or if we dont print the moulins we set this to an empty Int vector Int[]
    if !isempty(moulin_i) # if we had moulins, i.e. somewhere a non-zero in the ieb field 
        moulin_vals = Observable(selectdim(hist, ndims(hist), 1)[moulin_i, j])
        scatter!(ax, moulin_i, moulin_vals; color = :red, markersize = 8) # sets the red dot scatter point to signal in the plotting where each moulin was
        record(fig, filename, 1:ntimes; framerate = 20) do idx
            data[] = selectdim(hist, ndims(hist), idx)[:, j] # auto-updates the fig since data was wrapped in the Makie Observable struct
            moulin_vals[] = selectdim(hist, ndims(hist), idx)[moulin_i, j]
            ax.title = "t_iter = $(tracked_times[idx])"
        end
    else
        record(fig, filename, 1:ntimes; framerate = 20) do idx
            data[] = selectdim(hist, ndims(hist), idx)[:, j]
            ax.title = "t_iter = $(tracked_times[idx])"
        end
    end
end

"""
$(TYPEDSIGNATURES)

Renders an animated `.mp4` heatmap of a tracked field's full 2D time history `hist`, the 2D
counterpart of [`make_mp4_mid`](@ref) above (same arguments otherwise). `framerate` controls
playback speed only -- every frame in `hist` is still rendered, none are dropped. `title`, if
given, is set once and left static instead of the default per-frame `"t_iter = ..."` label.
`colormap` is passed straight through to `heatmap!`, e.g. `:inferno` for a black-background/
yellow-high look matching a `nan_color = :transparent` snapshot heatmap using the same colormap.
"""
function make_mp4_2d(hist::AbstractArray, tracked_times, moulin_ij; filename, show_moulins::Bool = true, framerate::Int = 20, title::Union{Nothing, String} = nothing, colormap = :viridis)
    ntimes = size(hist, ndims(hist))
    vmin   = minimum(hist)
    vmax   = maximum(hist)
    fig    = Figure(size = (1000, 800))
    ax     = Axis(fig[1, 1])
    data   = Observable(selectdim(hist, ndims(hist), 1))
    hm     = heatmap!(ax, data; colorrange = (vmin, vmax), colormap = colormap)
    Colorbar(fig[1, 2], hm)

    if show_moulins
        mx = Float32[mi for (mi, mj) in moulin_ij]
        my = Float32[mj for (mi, mj) in moulin_ij]
        scatter!(ax, mx, my; color = :red, markersize = 8)
    end

    title !== nothing && (ax.title = title)

    record(fig, filename, 1:ntimes; framerate = framerate) do idx
        data[] = selectdim(hist, ndims(hist), idx)
        title === nothing && (ax.title = "t_iter = $(tracked_times[idx])")
    end
end

"""
$(TYPEDSIGNATURES)

Convenience wrapper: [`make_mp4_mid`](@ref) directly off a [`LiveObserver`](@ref)'s `name`d
tracked field, instead of manually unpacking `obs.history[name]`/`obs.tracked_times`.
"""
make_mp4_mid(obs::LiveObserver, name::String, j, moulin_ij; filename, show_moulins::Bool = true) =
    make_mp4_mid(obs.history[name], obs.tracked_times, j, moulin_ij; filename = filename, show_moulins = show_moulins)

"""
$(TYPEDSIGNATURES)

Convenience wrapper: [`make_mp4_2d`](@ref) directly off a [`LiveObserver`](@ref)'s `name`d tracked
field.
"""
make_mp4_2d(obs::LiveObserver, name::String, moulin_ij; filename, show_moulins::Bool = true) =
    make_mp4_2d(obs.history[name], obs.tracked_times, moulin_ij; filename = filename, show_moulins = show_moulins)

"""
$(TYPEDSIGNATURES)

Extracts `(i,j)` moulin locations from `state.ieb`'s nonzero entries for [`make_mp4_mid`](@ref)/[`make_mp4_2d`](@ref).
"""
function get_moulin_ij(state::State)
    idxs = findall(!iszero, Array(state.ieb))
    return [(I[1], I[2]) for I in idxs]
end
