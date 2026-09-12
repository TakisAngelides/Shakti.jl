# =============================================================================
# FROZEN_BED transitions
# =============================================================================
#
# freeze_cells!/thaw_cells! are the only supported way to move cells to/from
# FROZEN_BED at runtime (e.g. from an external ice-thermal-state coupling,
# not yet wired up -- see mask.jl's FROZEN_BED docstring). Both take a
# same-shape Bool mask (same array type/backend as State's own fields --
# vectorized/broadcast throughout, no scalar getindex/setindex, so this stays
# GPU-safe) rather than a per-cell scalar API, since the real use case is
# freezing/thawing a whole region at once, not one cell at a time.

"""
$(TYPEDSIGNATURES)

Freezes every cell where `freeze_mask` is `true` and currently `GROUNDED` to
[`FROZEN_BED`](@ref) (cells `freeze_mask` marks that are already something else -- `OCEAN`/
`LAND`/`OTHER_BASIN`/already `FROZEN_BED` -- are left untouched, so passing an overly broad mask
is harmless, not an error).

Sets `b:=0` and `pw:=0` on those cells, then refreshes everything that depends on `s.mask`/`s.b`/
`s.pw`: `s.h` via [`compute_h!`](@ref) (`pw=0` gives `h=zb` exactly), `s.K` via
[`compute_K!`](@ref) (`b=0` gives `K=0` exactly), `s.valid_x`/`s.valid_y` via
[`compute_face_masks!`](@ref), `s.ub_x`/`s.ub_y` via [`apply_mask_to_sliding!`](@ref), and
`s.N` via [`compute_N!`](@ref) -- which, per `FROZEN_BED`'s own convention, evaluates to `po - 0
= po` exactly there: full overburden, no water pressure. `s.dpwdx`/`s.dpwdy` are left stale here
deliberately -- they're recomputed from the current `h`/`pw` at the top of every Picard iteration
during the next [`step_h!`](@ref) anyway, so refreshing them now would just be redone.

Not yet wired to any automatic per-timestep driver -- call this directly with whatever
`freeze_mask` your own driving logic computes (e.g. from an ice-thermal-state field).
"""
function freeze_cells!(s::State, p::ModelParameters, freeze_mask::AbstractMatrix{Bool})
    do_freeze = freeze_mask .& (s.mask .== GROUNDED)
    @. s.mask = ifelse(do_freeze, FROZEN_BED, s.mask)
    @. s.b = ifelse(do_freeze, zero(eltype(s.b)), s.b)
    @. s.pw = ifelse(do_freeze, zero(eltype(s.pw)), s.pw)
    compute_h!(s, p)
    compute_K!(s, p)
    compute_face_masks!(s)
    apply_mask_to_sliding!(s)
    compute_N!(s, p)
    return s
end

"""
$(TYPEDSIGNATURES)

Thaws every cell where `thaw_mask` is `true` and currently [`FROZEN_BED`](@ref) back to
`GROUNDED` (same "only touches cells actually in the expected starting category" behavior as
[`freeze_cells!`](@ref)).

Reseeds `b:=p.b_min` (a cell can't re-enter the dynamic assembly at literal `b=0` -- see
`FROZEN_BED`'s own docstring for why -- `compute_b!`'s usual kernels take over evolving it
normally from here on). `pw`/`h` are deliberately left untouched: a thawing cell starts from
whatever it was while frozen (`pw=0`/dry/high-`N` if it was actually frozen via
[`freeze_cells!`](@ref)) and the head equation evolves it from there on the next solve -- no
water is magically added. Refreshes `s.K`/`s.valid_x`/`s.valid_y`/`s.ub_x`/`s.ub_y`/`s.N` the same
way [`freeze_cells!`](@ref) does.
"""
function thaw_cells!(s::State, p::ModelParameters, thaw_mask::AbstractMatrix{Bool})
    do_thaw = thaw_mask .& (s.mask .== FROZEN_BED)
    @. s.mask = ifelse(do_thaw, GROUNDED, s.mask)
    @. s.b = ifelse(do_thaw, p.b_min, s.b)
    compute_K!(s, p)
    compute_face_masks!(s)
    apply_mask_to_sliding!(s)
    compute_N!(s, p)
    return s
end
