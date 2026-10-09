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
moulin locations on row `j` as red markers. Requires `using CairoMakie` (weak dependency, see `ext/ShaktiCairoMakieExt.jl`); uses CairoMakie's software rasterizer, so this works
on headless HPC nodes as well as locally.
"""
function make_mp4_mid end # methods for plain arrays live in ext/ShaktiCairoMakieExt.jl (load CairoMakie to enable)

"""
$(TYPEDSIGNATURES)

Renders an animated `.mp4` heatmap of a tracked field's full 2D time history `hist`, the 2D
counterpart of [`make_mp4_mid`](@ref) above (same arguments otherwise). `framerate` controls
playback speed only -- every frame in `hist` is still rendered, none are dropped. `title`, if
given, is set once and left static instead of the default per-frame `"t_iter = ..."` label.
`colormap` is passed straight through to `heatmap!`, e.g. `:inferno` for a black-background/
yellow-high look matching a `nan_color = :transparent` snapshot heatmap using the same colormap.
"""
function make_mp4_2d end # methods for plain arrays live in ext/ShaktiCairoMakieExt.jl (load CairoMakie to enable)

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
