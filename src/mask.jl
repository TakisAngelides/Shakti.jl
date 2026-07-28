# =============================================================================
# Mask conventions for State.mask
# =============================================================================
#
# State.mask is allocated with @fill(0, ...) in state.jl, so -- like valid_x/
# valid_y below -- it actually ends up float-valued (0.0/1.0/2.0/3.0), not a
# true Int array; @fill coerces to the backend's configured floattype
# regardless of the literal fill value's type. The constants below are
# therefore declared as Float64 literals too (not Int): comparisons like
# `mask[i,j] != OTHER_BASIN` are Float-Float, which compiles to a plain
# float comparison. Julia's Float-Int comparison isn't free even for small
# compile-time constants -- it has to guard against precision loss for
# large integers, which shows up as extra sitofp/fptosi round-trip
# instructions -- so keeping these as floats avoids that cost in these
# comparisons, which run in some of the hottest kernels in the solver
# (linear_solver.jl, k_face_scheme.jl, gap_height.jl).

"""
Mask value: dynamic hydrology solved here (Picard/Poisson elliptic solve + gap-height evolution).
"""
const GROUNDED = 0.0

"""
Mask value: Dirichlet boundary, `pw = p_atm - rho_sw*g*min(zb, 0)` (`zb` = bedrock elevation
relative to sea level, positive up, so a marine bed at `zb < 0` gets the correct hydrostatic
pressure at depth `-zb`; `rho_sw` is seawater density, since this is the ocean's own hydrostatic
pressure rather than the subglacial drainage system's).
"""
const OCEAN = 1.0

"""
Mask value: Dirichlet boundary, `pw = p_atm` (`0.0` by default).
"""
const LAND = 2.0

"""
Mask value: not solved here; frozen row. Any `GROUNDED` neighbour treats the shared face as
zero-flux (Neumann), and any face-based quantity (`dhdx`, `dpwdx`, `q_x`, ...) touching this cell
is zeroed.
"""
const OTHER_BASIN = 3.0

# =============================================================================
# Face-validity bookkeeping
# =============================================================================
#
# valid_x/valid_y hold float 1.0 (valid) / 0.0 (invalid), not Bool -- @fill
# always coerces to the backend's configured floattype (see state.jl), so
# there's no genuine Bool array available here. They're used as multiplicative
# masks in fields_gradients.jl (e.g. `dhdx[...] * valid_x[...]`), which is
# exactly what a 1.0/0.0 float wants to be used for anyway.
#
# A face is invalid iff either cell it connects is OTHER_BASIN: that cell's
# hydrology isn't solved here, so any gradient computed across that face would
# spuriously reflect a frozen, non-evolving neighbour value rather than a real
# head/pressure difference. LAND and OCEAN faces are left valid, since those
# are genuine (Dirichlet) drainage boundaries where a real flux is physically
# meaningful. Outer boundary faces (ix==1/end for x, iy==1/end for y) are left
# at 1.0: compute_dhdx! etc. never write those entries anyway (their update
# ranges are 2:end-1), so they stay at their initialized value of zero.

@parallel_indices (ix, iy) function compute_valid_x_kernel!(valid_x, mask)
    if ix <= size(valid_x, 1) && iy <= size(valid_x, 2)
        if ix > 1 && ix < size(valid_x, 1)
            valid = (mask[ix-1, iy] != OTHER_BASIN) && (mask[ix, iy] != OTHER_BASIN) # both cells touching the face must not be OTHER_BASIN
            valid_x[ix, iy] = valid ? one(eltype(valid_x)) : zero(eltype(valid_x))
        else
            valid_x[ix, iy] = one(eltype(valid_x))
        end
    end
    return
end

@parallel_indices (ix, iy) function compute_valid_y_kernel!(valid_y, mask)
    if ix <= size(valid_y, 1) && iy <= size(valid_y, 2)
        if iy > 1 && iy < size(valid_y, 2)
            valid = (mask[ix, iy-1] != OTHER_BASIN) && (mask[ix, iy] != OTHER_BASIN) # both cells touching the face must not be OTHER_BASIN
            valid_y[ix, iy] = valid ? one(eltype(valid_y)) : zero(eltype(valid_y))
        else
            valid_y[ix, iy] = one(eltype(valid_y))
        end
    end
    return
end

"""
$(TYPEDSIGNATURES)

Recomputes `s.valid_x`/`s.valid_y` from `s.mask`. Must be called (directly, or via
[`set_initial_conditions!`](@ref)) any time `s.mask` changes.

# Notes

A face is invalid iff either cell it connects is `OTHER_BASIN`: that cell's hydrology isn't
solved, so any gradient computed across that face would spuriously reflect a frozen,
non-evolving neighbour value rather than a real head/pressure difference. `LAND` and `OCEAN`
faces are left valid, since those are genuine (Dirichlet) drainage boundaries where a real flux
is physically meaningful.
"""
function compute_face_masks!(s::State)
    @parallel compute_valid_x_kernel!(s.valid_x, s.mask)
    @parallel compute_valid_y_kernel!(s.valid_y, s.mask)
    return s
end

# =============================================================================
# Sliding-velocity mask bookkeeping
# =============================================================================
#
# Zeroes ub_x/ub_y on any face touching a non-GROUNDED cell, consistent with
# the mask: sliding velocity is only physically meaningful where grounded ice
# is actually sliding on a bed with an evolving hydraulic system. We cannot have melting
# happening through the term taub ⋅ ub on a grid cell that is OTHER_BASIN, or for the face
# between GROUNDED and OTHER_BASIN to contribute to that melting. This will stay consistent with
# the Neumann conditions we impose at the point we solve the elliptic equation for h. 

@parallel_indices (ix, iy) function apply_mask_to_sliding_x_kernel!(ub_x, mask)
    if ix <= size(ub_x, 1) && iy <= size(ub_x, 2)
        # x-faces: interior faces need both neighbours grounded; the two domain
        # boundary faces (ix=1, ix=nx+1) have only one neighbour, so always zeroed.
        if ix == 1 || ix == size(ub_x, 1) || !(mask[ix-1, iy] == GROUNDED && mask[ix, iy] == GROUNDED)
            ub_x[ix, iy] = zero(eltype(ub_x))
        end
    end
    return
end

@parallel_indices (ix, iy) function apply_mask_to_sliding_y_kernel!(ub_y, mask)
    if ix <= size(ub_y, 1) && iy <= size(ub_y, 2)
        # y-faces: same as above
        if iy == 1 || iy == size(ub_y, 2) || !(mask[ix, iy-1] == GROUNDED && mask[ix, iy] == GROUNDED)
            ub_y[ix, iy] = zero(eltype(ub_y))
        end
    end
    return
end

"""
$(TYPEDSIGNATURES)

Zeroes `s.ub_x`/`s.ub_y` on any face touching a non-`GROUNDED` cell: sliding velocity is only
physically meaningful where grounded ice is actually sliding on a bed with an evolving hydraulic
system, so a face touching `OTHER_BASIN` (or a domain boundary) cannot contribute melting via the
`taub . ub` term, consistent with the Neumann condition imposed on the elliptic head solve there.
"""
function apply_mask_to_sliding!(s::State)
    @parallel apply_mask_to_sliding_x_kernel!(s.ub_x, s.mask)
    @parallel apply_mask_to_sliding_y_kernel!(s.ub_y, s.mask)
    return s
end