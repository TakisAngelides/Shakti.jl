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

For a threshold-based driver on a basal-temperature field, see [`update_frozen_mask!`](@ref),
which calls this (and [`thaw_cells!`](@ref)) for you.

# Limitation

Freezing is a cutoff, not a phase-change model: any water still held in a freezing cell (`b`) is
discarded, not refrozen onto the ice base, and no latent heat is applied. Gradual refreezing is
the job of the energy balance already in [`compute_mdot!`](@ref) (the `q_T` conductive-loss term
drives `mdot < 0` and shrinks `b` on cold ice), so a more negative `T_freeze` lets that dry the
cell out first and leaves less to discard. [`update_frozen_mask!`](@ref) reports the discarded
amount so the size of this mass loss can be checked.
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

"""
$(TYPEDSIGNATURES)

Updates `s.mask` between `GROUNDED` and `FROZEN_BED` from the basal temperature relative to
pressure melting `T_prime_b` (K, an array the same shape as `s.mask`, e.g. Yelmo's `T_prime_b`),
using `p.T_freeze` and `p.T_hysteresis`:

  - a `GROUNDED` cell with `T_prime_b < T_freeze` freezes ([`freeze_cells!`](@ref));
  - a `FROZEN_BED` cell with `T_prime_b >= T_freeze + T_hysteresis` thaws ([`thaw_cells!`](@ref));
  - anything else keeps its current state (this band is the hysteresis that prevents flicker).

Throws an `ArgumentError` (before changing anything) if cells are due to thaw while `p.b_min <= 0`:
thawed cells restart at `b_min`, and a thawed region at `b = 0` makes the head system singular (see
[`thaw_cells!`](@ref)) -- silently so under [`CUDSSDirectSolver`](@ref), which does not check.

Cells that are not `GROUNDED`/`FROZEN_BED` (`OCEAN`/`LAND`/`OTHER_BASIN`) are never touched. The
refresh work in `freeze_cells!`/`thaw_cells!` only runs on calls where a transition actually
happens. Intended to be called whenever the basal temperature is refreshed (thermodynamic
timescale), not every hydrology step.

Returns `(n_frozen, n_thawed, discarded_b)`: the number of cells that changed each way and the sum
of the gap height `b` (m) over the freezing cells, i.e. the water thickness discarded by freezing
(multiply by `dx*dy` for a volume, see the limitation note in [`freeze_cells!`](@ref)).
"""
function update_frozen_mask!(s::State, p::ModelParameters, T_prime_b::AbstractMatrix)
    T_thaw = p.T_freeze + p.T_hysteresis
    freeze_mask = (s.mask .== GROUNDED)   .& (T_prime_b .< p.T_freeze)
    thaw_mask   = (s.mask .== FROZEN_BED) .& (T_prime_b .>= T_thaw)
    n_frozen = count(freeze_mask) # one reduction each (a device sync on GPU); called once per thermodynamic update, so negligible next to a solve
    n_thawed = count(thaw_mask)
    n_thawed > 0 && p.b_min <= 0 && throw(ArgumentError("update_frozen_mask! would thaw $n_thawed cell(s) but p.b_min = $(p.b_min) <= 0: thawed cells are reseeded at b_min, and a region reseeded at b = 0 gives all-zero matrix rows (K = 0 and no creep-closure term), i.e. a singular system -- CholeskyDirectSolver throws PosDefException and CUDSSDirectSolver silently returns garbage. Build ModelParameters with b_min > 0 (e.g. 1e-3)."))
    discarded_b = zero(eltype(s.b))
    if n_frozen > 0
        discarded_b = sum(s.b .* freeze_mask) # read before freeze_cells! zeroes b
        freeze_cells!(s, p, freeze_mask)
    end
    n_thawed > 0 && thaw_cells!(s, p, thaw_mask)
    return (n_frozen = n_frozen, n_thawed = n_thawed, discarded_b = discarded_b)
end
