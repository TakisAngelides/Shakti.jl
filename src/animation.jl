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

# Convenience wrappers so you can call these directly off a LiveObserver
# instead of manually unpacking obs.history[name]/obs.tracked_times every time.
make_mp4_mid(obs::LiveObserver, name::String, j, moulin_ij; filename, show_moulins::Bool = true) =
    make_mp4_mid(obs.history[name], obs.tracked_times, j, moulin_ij; filename = filename, show_moulins = show_moulins)

make_mp4_2d(obs::LiveObserver, name::String, moulin_ij; filename, show_moulins::Bool = true) =
    make_mp4_2d(obs.history[name], obs.tracked_times, moulin_ij; filename = filename, show_moulins = show_moulins)

# Extracts (i,j) moulin locations from state.ieb (nonzero entries), so you
# don't have to hand-build moulin_ij yourself.
#
# state.ieb may be GPU-resident (under a GPU backend): `findall` on it isn't
# guaranteed to be implemented/efficient for every GPU array type, and the
# comprehension below would need scalar getindex on its result either way --
# disallowed on GPU arrays. Array(...) brings it to the host first (a
# deliberate, one-off device->host transfer, same as observer.jl's
# Array{Float64}(...) conversion -- this is a post-processing/plotting
# convenience function, not a hot loop).
function get_moulin_ij(state::State)
    idxs = findall(!iszero, Array(state.ieb))
    return [(I[1], I[2]) for I in idxs]
end
