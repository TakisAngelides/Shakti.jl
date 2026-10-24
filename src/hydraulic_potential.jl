# =============================================================================
# Hydraulic potential: bed head or water-surface head
# =============================================================================
# Shakti's head is referenced to the bed, h = zb + pw/(rho_w g) (Sommers et al. 2018), so the potential
# that drives the flux, phi = rho_w g h, omits the thickness of the water layer. GlaDS-2 (Wells et al.
# 2026, Eq. 7) keeps it: phi = rho_w g (zb + h_w) + pw, with pw the pressure at the top of the flowing
# water and h_w its thickness. In Shakti's variables h_w = b in a filled cell, and in an unfilled cell
# (unfilled_cavities.jl) Shakti's head is already h = zb - (b - W) = zb + W - b, W the
# water depth. So in every regime
#     phi / (rho_w g) = h + b,
# and the only term missing from the flux is -K grad(b). [`SheetTopPotential`](@ref) adds it: the flux
# becomes q = -K grad(h + b), with b the gap at the start of the step (held fixed during the head solve).
# Under ConsistentGapCoupling the conductance takes the end-of-step gap while this jump keeps the
# start-of-step one: a lag of O(db per step) in the flux, so the water moved in a step is off by
# O(dt^2), the same order as the backward-Euler error of the step itself; it vanishes at steady state. The head h keeps its meaning (bed-referenced), so pw, N and the unfilled
# criterion are unchanged; only the driving gradient changes.
#
# Implementation: `State.db_x`/`db_y` hold the jump of b across each face, b[i] - b[i-1], on faces
# between two GROUNDED cells. On a face to an OCEAN/LAND boundary the boundary's side counts as 0 (its
# fixed head, the sea level or the bed at atmospheric pressure, is already the level of its water), so
# the jump is +-b of the grounded cell: the outlet is driven by h_D - (h + b), like every interior face
# by the water level, and the outflow-only faces open and close (compute_face_masks!, the outflow stops
# of limit_head_step!) at the same level. Without it an outlet face saw h_D - h only, underestimating the
# outflow by a fraction of order b/(h - h_D) where the margin is thin. Faces touching OTHER_BASIN/
# FROZEN_BED keep 0, so the frozen rows are untouched. The head gradients (`compute_dhdxy!`,
# field_gradients.jl) add them, so everything built from `dhdx`/`dhdy` -- face flux and Re, the upwind
# donor, the dissipation melt, SUHMO's D, the water budget -- sees the same potential, and the head rows
# (linear_solver.jl, `add_sheet_top_source!`) move the known part K*db to the right-hand side. Under
# [`BedPotential`](@ref) (legacy) the jumps stay zero and both places skip the addition, so results are
# bit-for-bit those of the original equations. SheetTopPotential is the default: the water's level is h + b in
# every regime, so grad(h) alone would miss the flow (and dissipation) along differences in gap size.

"""
$(TYPEDSIGNATURES)

Which hydraulic potential drives the water flux -- multiple dispatch on the concrete subtype
([`BedPotential`](@ref)/[`SheetTopPotential`](@ref)), chosen once by the caller of `Simulation`
(keyword `hydraulic_potential`), same idiom as [`AbstractCavityFilling`](@ref).
"""
abstract type AbstractHydraulicPotential end

"""
$(TYPEDSIGNATURES)

Legacy: the potential is `rho_w g zb + pw` (Sommers et al. 2018), the flux `q = -K grad(h)`. Bit-for-bit the
original equations; pass it explicitly (the default is [`SheetTopPotential`](@ref)).
"""
struct BedPotential <: AbstractHydraulicPotential end

"""
$(TYPEDSIGNATURES)

The potential includes the thickness of the flowing water, `rho_w g (zb + h_w) + pw` (GlaDS-2, Wells et
al. 2026, Eq. 7), i.e. `q = -K grad(h + b)` with the gap `b` held at its start-of-step value. The default.
See this file's header.
"""
struct SheetTopPotential <: AbstractHydraulicPotential end

# The gap a face sees on each side: the cell's own b if GROUNDED, 0 at an OCEAN/LAND boundary (whose fixed head is
# already the level of its water), nothing at all at OTHER_BASIN/FROZEN_BED (the face is closed, valid_x/valid_y).
@inline jump_side(m) = m == GROUNDED || m == OCEAN || m == LAND
@inline face_gap_jump(m1, m2, b1, b2) =
    (jump_side(m1) && jump_side(m2) && (m1 == GROUNDED || m2 == GROUNDED)) ?
        (m2 == GROUNDED ? b2 : zero(b2)) - (m1 == GROUNDED ? b1 : zero(b1)) : zero(b1)

@parallel_indices (ix, iy) function set_gap_jumps_kernel!(db_x, db_y, b, mask)
    nx, ny = size(b, 1), size(b, 2)
    if ix <= size(db_x, 1) && iy <= size(db_x, 2)
        db_x[ix, iy] = (ix > 1 && ix <= nx && iy <= ny) ? face_gap_jump(mask[ix-1, iy], mask[ix, iy], b[ix-1, iy], b[ix, iy]) : zero(eltype(b))
    end
    if ix <= size(db_y, 1) && iy <= size(db_y, 2)
        db_y[ix, iy] = (iy > 1 && iy <= ny && ix <= nx) ? face_gap_jump(mask[ix, iy-1], mask[ix, iy], b[ix, iy-1], b[ix, iy]) : zero(eltype(b))
    end
    return
end

"""
$(TYPEDSIGNATURES)

Sets the face jumps of the gap height `s.db_x`/`s.db_y` from the current `s.b` (called once per step,
before the head solve): `b[i] - b[i-1]` on faces between two `GROUNDED` cells, `+b` or `-b` of the grounded cell on a
face to an `OCEAN`/`LAND` boundary (whose side counts as 0: its fixed head is the level of its water), `0` elsewhere.
A no-op under [`BedPotential`](@ref), where they stay zero.
"""
update_gap_jumps!(s::State, ::BedPotential) = s
update_gap_jumps!(s::State, ::SheetTopPotential) = (@parallel set_gap_jumps_kernel!(s.db_x, s.db_y, s.b, s.mask); s)
