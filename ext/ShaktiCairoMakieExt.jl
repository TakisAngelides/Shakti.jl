# Package extension: animation back end, active once the user has loaded CairoMakie (`using CairoMakie`).
# Keeps CairoMakie (a very heavy load) out of the dependency chain of every plain `using Shakti`.
module ShaktiCairoMakieExt

using Shakti
using CairoMakie
using CairoMakie: Makie

import Shakti: make_mp4_mid, make_mp4_2d

function Shakti.make_mp4_mid(hist::AbstractArray, tracked_times, j, moulin_ij; filename, show_moulins::Bool = true)
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
        Makie.scatter!(ax, moulin_i, moulin_vals; color = :red, markersize = 8) # sets the red dot scatter point to signal in the plotting where each moulin was -- explicitly qualified: CUDSS.jl's SciMLBase dependency activates SciMLBaseMakieExt once Makie is also loaded, which makes the bare `scatter!` name ambiguous
        Makie.record(fig, filename, 1:ntimes; framerate = 20) do idx # explicitly qualified -- CUDA.jl's own `record` (stream/event recording) collides with Makie's once both are loaded
            data[] = selectdim(hist, ndims(hist), idx)[:, j] # auto-updates the fig since data was wrapped in the Makie Observable struct
            moulin_vals[] = selectdim(hist, ndims(hist), idx)[moulin_i, j]
            ax.title = "t_iter = $(tracked_times[idx])"
        end
    else
        Makie.record(fig, filename, 1:ntimes; framerate = 20) do idx
            data[] = selectdim(hist, ndims(hist), idx)[:, j]
            ax.title = "t_iter = $(tracked_times[idx])"
        end
    end
end

function Shakti.make_mp4_2d(hist::AbstractArray, tracked_times, moulin_ij; filename, show_moulins::Bool = true, framerate::Int = 20, title::Union{Nothing, String} = nothing, colormap = :viridis)
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
        Makie.scatter!(ax, mx, my; color = :red, markersize = 8) # explicitly qualified -- see the other scatter! call site's comment above for why
    end

    title !== nothing && (ax.title = title)

    Makie.record(fig, filename, 1:ntimes; framerate = framerate) do idx # explicitly qualified -- see make_mp4_mid's own comment for why
        data[] = selectdim(hist, ndims(hist), idx)
        title === nothing && (ax.title = "t_iter = $(tracked_times[idx])")
    end
end

end # module
