# =============================================================================
# Animation
# =============================================================================
#
# Turns a LiveObserver's in-RAM history into .mp4 animations. Rendered with
# CairoMakie (software/Cairo rasterizer, via record()'s backend-agnostic
# colorbuffer()) rather than GLMakie -- no OpenGL/display needed, so this
# works on headless HPC nodes as well as locally. LiveObserver.history[name]
# is a single preallocated array of shape (field's own shape...,
# length(tracked_times)) -- time is the last dimension -- unlike src_old's
# Observer.history[field], which was a Vector{Array} (one array per saved
# frame). Frames are selected here with selectdim(..., ndims(hist), idx)
# instead of the old Dict/Vector indexing.
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
    ntimes = size(hist, ndims(hist))
    ymin   = minimum(hist)
    ymax   = maximum(hist)
    x      = 1:size(hist, 1)
    data   = Observable(selectdim(hist, ndims(hist), 1)[:, j])
    fig    = Figure(size = (1000, 800))
    ax     = Axis(fig[1, 1]; xlabel = "i", ylabel = "Value", limits = (nothing, (ymin, ymax)))
    lines!(ax, x, data)

    moulin_i = show_moulins ? [mi for (mi, mj) in moulin_ij if mj == j] : Int[]
    if !isempty(moulin_i)
        moulin_vals = Observable(selectdim(hist, ndims(hist), 1)[moulin_i, j])
        scatter!(ax, moulin_i, moulin_vals; color = :red, markersize = 8)
        record(fig, filename, 1:ntimes; framerate = 20) do idx
            data[] = selectdim(hist, ndims(hist), idx)[:, j]
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
counterpart of [`make_mp4_mid`](@ref) above (same arguments otherwise).
"""
function make_mp4_2d(hist::AbstractArray, tracked_times, moulin_ij; filename, show_moulins::Bool = true)
    ntimes = size(hist, ndims(hist))
    vmin   = minimum(hist)
    vmax   = maximum(hist)
    fig    = Figure(size = (1000, 800))
    ax     = Axis(fig[1, 1])
    data   = Observable(selectdim(hist, ndims(hist), 1))
    hm     = heatmap!(ax, data; colorrange = (vmin, vmax))
    Colorbar(fig[1, 2], hm)

    if show_moulins
        mx = Float32[mi for (mi, mj) in moulin_ij]
        my = Float32[mj for (mi, mj) in moulin_ij]
        scatter!(ax, mx, my; color = :red, markersize = 8)
    end

    record(fig, filename, 1:ntimes; framerate = 20) do idx
        data[] = selectdim(hist, ndims(hist), idx)
        ax.title = "t_iter = $(tracked_times[idx])"
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

Extracts `(i,j)` moulin locations from `state.ieb`'s nonzero entries, so you don't have to
hand-build `moulin_ij` yourself for [`make_mp4_mid`](@ref)/[`make_mp4_2d`](@ref).
"""
function get_moulin_ij(state::State)
    idxs = findall(!iszero, Array(state.ieb))
    return [(I[1], I[2]) for I in idxs]
end
