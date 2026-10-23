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

Refreshes the fields that are lagged functions of `s.b` -- `s.beta` ([`compute_beta!`](@ref)) and `s.lc`
([`compute_lc!`](@ref)) -- which [`step_b!`](@ref) and [`set_initial_conditions!`](@ref) always update right
after `b` changes. The schemes are chosen from `p` exactly as those two do (`StandardCreep` iff `p.b_c == 0`,
`NoOpenBySliding` iff `p.br == 0`).
"""
function refresh_b_dependents!(s::State, p::ModelParameters)
    compute_beta!(s, p, iszero(p.br) ? NoOpenBySliding() : WithOpenBySliding())
    compute_lc!(s, p, iszero(p.b_c) ? StandardCreep() : CreepCutoff())
    return s
end

"""
$(TYPEDSIGNATURES)

Freezes every cell where `freeze_mask` is `true` and currently `GROUNDED` to
[`FROZEN_BED`](@ref) (cells `freeze_mask` marks that are already something else -- `OCEAN`/
`LAND`/`OTHER_BASIN`/already `FROZEN_BED` -- are left untouched, so passing an overly broad mask
is harmless, not an error).

Sets `b:=0` and `pw:=0` on those cells, and zeroes the water fields (`b_empty`, `b_uplift`, `b_dry`, `w_old`), then refreshes everything that depends on `s.mask`/`s.b`/
`s.pw`: `s.h = zb` on those cells (`pw = 0`; the other cells keep their head), the cell-centred `s.K`
(`b=0` gives `K=0` exactly; the face transmissivities are rebuilt by the next refresh), `s.valid_x`/`s.valid_y` via
[`compute_face_masks!`](@ref), and
`s.N` via [`compute_N!`](@ref) -- which, per `FROZEN_BED`'s own convention, evaluates to `po - 0
= po` exactly there: full overburden, no water pressure. `s.dpwdx`/`s.dpwdy` are left stale here
deliberately -- they're recomputed from the current `h`/`pw` at the top of every Picard iteration
during the next [`step_h!`](@ref) anyway, so refreshing them now would just be redone.

For a threshold-based driver on a basal-temperature field, see [`update_frozen_mask!`](@ref),
which calls this (and [`thaw_cells!`](@ref)) for you. These `State` methods know none of the simulation's options (cell
`N` clamps, outflow-only faces, face scheme); the `Simulation` methods, `freeze_cells!(sim, freeze_mask)` and likewise for
[`thaw_cells!`](@ref), [`set_mask!`](@ref) and [`update_frozen_mask!`](@ref), refresh the derived fields with them
afterwards ([`refresh_state!`](@ref)).

# Limitation

Freezing is a cutoff, not a phase-change model: any water still held in a freezing cell (the filled gap and any uplift layer) is
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
    clear_water!(s, do_freeze)
    @. s.pw = ifelse(do_freeze, zero(eltype(s.pw)), s.pw)
    refresh_b_dependents!(s, p) # lc/beta are lagged fields derived from b (normally refreshed in step_b!): keep them consistent with the b just set
    @. s.h = ifelse(do_freeze, s.zb, s.h) # pw = 0; the other cells keep their head (recomputing it from pw dropped the empty depth and uplift layer it holds)
    compute_cell_K!(s, p) # the face transmissivities are rebuilt by the next refresh (compute_face_flux!)
    compute_face_masks!(s)
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
normally from here on), marked empty: `b_empty = b_min` for [`UnfilledCavities`](@ref) and `b_dry = b_min`
for the dry floor (`dry_floor`); the option that does not own a field zeroes it at the next step. So the
cell starts with no water, except under the filled sheet without the dry floor, whose water is `b` itself
(`b_min` per thawed cell). `pw`/`h` are deliberately left untouched: a thawing cell starts from
whatever it was while frozen (`pw=0`/dry/high-`N` if it was actually frozen via
[`freeze_cells!`](@ref)) and the head equation evolves it from there on the next solve. Refreshes `s.K`/`s.valid_x`/`s.valid_y`/`s.N` the same
way [`freeze_cells!`](@ref) does.
"""
function thaw_cells!(s::State, p::ModelParameters, thaw_mask::AbstractMatrix{Bool})
    do_thaw = thaw_mask .& (s.mask .== FROZEN_BED)
    @. s.mask = ifelse(do_thaw, GROUNDED, s.mask)
    seed_empty!(s, p, do_thaw)
    refresh_b_dependents!(s, p) # without this a thawed cell keeps lc = 0 (from its frozen b = 0) until the next step_b!, i.e. no creep-closure term in the first head solve
    compute_cell_K!(s, p) # the face transmissivities are rebuilt by the next refresh (compute_face_flux!)
    compute_face_masks!(s)
    compute_N!(s, p)
    return s
end

# The water a cell holds, as w_old counts it (set_cavity_storage!): the filled gap less the dry floor, plus the uplift
# layer. Each term is zero under the options that do not own it, so this is the water under every sheet.
held_water(s::State, cells) = sum((max.(s.b .- s.b_empty .- s.b_dry, zero(eltype(s.b))) .+ s.b_uplift) .* cells)

# A cell leaving the dynamic sheet keeps no gap and no water.
function clear_water!(s::State, cells)
    z = zero(eltype(s.b))
    @. s.b = ifelse(cells, z, s.b)
    @. s.b_empty = ifelse(cells, z, s.b_empty)
    @. s.b_uplift = ifelse(cells, z, s.b_uplift)
    @. s.b_dry = ifelse(cells, z, s.b_dry)
    @. s.w_old = ifelse(cells, z, s.w_old)
    @. s.W = ifelse(cells, z, s.W)
    return s
end

# A cell entering the dynamic sheet gets the gap b_min but no water: the gap is marked empty both as an unfilled
# cavity (b_empty) and as the dry floor (b_dry). The option that does not own a field zeroes it at the next step
# (prepare_step_inputs!), and the one that does keeps the cell dry. Only the filled sheet without the dry floor,
# whose water is b itself, still starts these cells with b_min of water.
function seed_empty!(s::State, p::ModelParameters, cells)
    z = zero(eltype(s.b))
    @. s.b = ifelse(cells, p.b_min, s.b)
    @. s.b_empty = ifelse(cells, p.b_min, s.b_empty)
    @. s.b_dry = ifelse(cells, p.b_min, s.b_dry)
    @. s.b_uplift = ifelse(cells, z, s.b_uplift)
    @. s.W = ifelse(cells, z, s.W)
    return s
end

@inline drains_to(m) = m == GROUNDED || m == LAND || m == OCEAN

@parallel_indices (ix, iy) function isolated_cells_kernel!(iso, mask)
    nx, ny = size(mask, 1), size(mask, 2)
    if ix <= nx && iy <= ny
        iso[ix, iy] = mask[ix, iy] == GROUNDED &&
                      !((ix > 1 && drains_to(mask[ix-1, iy])) || (ix < nx && drains_to(mask[ix+1, iy])) ||
                        (iy > 1 && drains_to(mask[ix, iy-1])) || (iy < ny && drains_to(mask[ix, iy+1])))
    end
    return
end

"""
$(TYPEDSIGNATURES)

Freezes ([`freeze_cells!`](@ref)) every `GROUNDED` cell none of whose four neighbours is `GROUNDED`,
`LAND` or `OCEAN`: all its faces are closed (frozen, `OTHER_BASIN` or the domain edge), so its water
has nowhere to go and, where its creep term vanishes (e.g. `N = 0` under `N_min = 0`), its head-equation
row is all zero and the solve fails. Freezing such a cell can isolate no other one (an isolated cell has
no `GROUNDED` neighbour), so one pass suffices. One stencil kernel and a count, run where the mask
changes, not per step. Returns the number of cells frozen.
"""
function freeze_isolated!(s::State, p::ModelParameters)
    iso = similar(s.mask, Bool)
    @parallel (1:size(s.mask, 1), 1:size(s.mask, 2)) isolated_cells_kernel!(iso, s.mask)
    n_iso = count(iso)
    n_iso > 0 && freeze_cells!(s, p, iso)
    return n_iso
end

"""
$(TYPEDSIGNATURES)

Updates `s.mask` between `GROUNDED` and `FROZEN_BED` from the basal temperature relative to
pressure melting `T_prime_b` (K, an array the same shape as `s.mask`, e.g. Yelmo's `T_prime_b`),
using `p.T_freeze` and `p.T_hysteresis`:

  - a `GROUNDED` cell with `T_prime_b < T_freeze` freezes ([`freeze_cells!`](@ref));
  - a `FROZEN_BED` cell with `T_prime_b >= T_freeze + T_hysteresis` thaws ([`thaw_cells!`](@ref));
  - anything else keeps its current state (this band is the hysteresis that prevents flicker).

Thawed cells restart at `p.b_min` (see [`thaw_cells!`](@ref)). With `b_min = 0` that is `b = 0`, and a
cell whose own `b` and all neighbours' `b` are zero has an all-zero matrix row -- the same singular
system a `GROUNDED` cell can reach on its own once `b` creeps to zero in a zero-`b` patch
([`CholeskyDirectSolver`](@ref) throws `PosDefException`; [`CUDSSDirectSolver`](@ref) does not check
and silently returns garbage). A thawing block makes that certain for its interior, so use `b_min > 0`
(e.g. `1e-3`) in runs that thaw regions.

Cells that are not `GROUNDED`/`FROZEN_BED` (`OCEAN`/`LAND`/`OTHER_BASIN`) are never touched. The
refresh work in `freeze_cells!`/`thaw_cells!` only runs on calls where a transition actually
happens. Intended to be called whenever the basal temperature is refreshed (thermodynamic
timescale), not every hydrology step.

Returns `(n_frozen, n_thawed, discarded_b, n_isolated)`: the number of cells that changed each way, the sum
of the water depth (m; the filled gap `b - b_empty - b_dry` plus the uplift layer `b_uplift`) over the freezing cells, i.e.
the water thickness discarded by freezing
(multiply by `dx*dy` for a volume, see the limitation note in [`freeze_cells!`](@ref)).
"""
function update_frozen_mask!(s::State, p::ModelParameters, T_prime_b::AbstractMatrix)
    T_thaw = p.T_freeze + p.T_hysteresis
    freeze_mask = (s.mask .== GROUNDED)   .& (T_prime_b .< p.T_freeze)
    thaw_mask   = (s.mask .== FROZEN_BED) .& (T_prime_b .>= T_thaw)
    n_frozen = count(freeze_mask) # one reduction each (a device sync on GPU); called once per thermodynamic update, so negligible next to a solve
    n_thawed = count(thaw_mask)
    discarded_b = zero(eltype(s.b))
    if n_frozen > 0
        discarded_b = held_water(s, freeze_mask) # read before freeze_cells! zeroes it
        freeze_cells!(s, p, freeze_mask)
    end
    n_thawed > 0 && thaw_cells!(s, p, thaw_mask)
    n_isolated = freeze_isolated!(s, p)   # freezing can cut a GROUNDED cell off from every drainage path
    return (n_frozen = n_frozen, n_thawed = n_thawed, discarded_b = discarded_b, n_isolated = n_isolated)
end


"""
$(TYPEDSIGNATURES)

Reclassifies cells to a host model's current mask `new_mask` (codes `GROUNDED`, `OCEAN`, `LAND`,
`OTHER_BASIN`; e.g. where the ice-sheet model has grounded ice it solves), keeping `FROZEN_BED`
cells frozen where the host still has them grounded (the frozen bed is handled by
[`update_frozen_mask!`](@ref) after this). Call it at each coupling step, before the hydrology step,
so Shakti never solves where the host has no grounded ice and never ignores where it does.

- A cell that stops being `GROUNDED` loses its gap and water (discarded, returned as in [`update_frozen_mask!`](@ref)) and takes the
  prescribed `pw` of its new category ([`apply_boundary_pw!`](@ref)).
- A cell that becomes `GROUNDED` is seeded at `b = p.b_min` with no water, as a thawing cell is; its `pw` (the
  Dirichlet value of its old category) is the initial guess of the next head solve.

Then refreshes the mask-dependent fields as [`freeze_cells!`](@ref) does. Returns
`(n_on, n_off, discarded_b, n_isolated)`.
"""
function set_mask!(s::State, p::ModelParameters, new_mask::AbstractMatrix)
    nm = similar(s.mask)
    copyto!(nm, new_mask)
    @. nm = ifelse((s.mask == FROZEN_BED) & (nm == GROUNDED), FROZEN_BED, nm)
    on  = (nm .== GROUNDED) .& (s.mask .!= GROUNDED)
    off = (s.mask .== GROUNDED) .& (nm .!= GROUNDED)
    n_on, n_off = count(on), count(off)
    discarded_b = zero(eltype(s.b))
    (n_on == 0 && n_off == 0 && s.mask == nm) && return (n_on = 0, n_off = 0, discarded_b = discarded_b, n_isolated = 0)
    discarded_b = held_water(s, off)
    s.mask .= nm
    clear_water!(s, off)
    seed_empty!(s, p, on)
    apply_boundary_pw!(s, p)
    refresh_b_dependents!(s, p)
    rg = p.rho_w * p.g
    @. s.h = ifelse(on | (s.mask != GROUNDED), s.pw / rg + s.zb, s.h) # the cells that changed and the boundaries from pw; a grounded cell keeps its head, with the empty depth or uplift layer it holds
    compute_cell_K!(s, p) # the face transmissivities are rebuilt by the next refresh (compute_face_flux!)
    compute_face_masks!(s)
    compute_N!(s, p)
    n_isolated = freeze_isolated!(s, p)   # kept FROZEN_BED cells can cut a new GROUNDED cell off from every drainage path
    return (n_on = n_on, n_off = n_off, discarded_b = discarded_b, n_isolated = n_isolated)
end
