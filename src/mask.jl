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

"""
Mask value: a genuinely frozen bed -- no water, so no gap height (`b=0`) and no water pressure
(`pw=0`, giving `N=po` exactly; see [`compute_N!`](@ref)). Otherwise handled identically to
[`OTHER_BASIN`](@ref): not solved here (frozen Dirichlet row in the elliptic solve), any
`GROUNDED` neighbour treats the shared face as zero-flux, and any face-based quantity touching
this cell is zeroed. Distinct from `OTHER_BASIN` (which is a domain-restriction category -- real
ice just outside the region being solved, `pw` held at whatever `p_atm` happens to be) precisely
in that `N` is computed here rather than reported as `0`, and its own `pw` is pinned to exactly
`0` regardless of `p_atm`. Transition a cell to/from `FROZEN_BED` at runtime with
[`freeze_cells!`](@ref)/[`thaw_cells!`](@ref) (`frozen_bed.jl`), or let
[`update_frozen_mask!`](@ref) drive them from a basal-temperature field and the `T_freeze`/
`T_hysteresis` parameters.
"""
const FROZEN_BED = 4.0

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
# A face is invalid iff either cell it connects is OTHER_BASIN or FROZEN_BED: that cell's
# hydrology isn't solved here, so any gradient computed across that face would
# spuriously reflect a frozen, non-evolving neighbour value rather than a real
# head/pressure difference. LAND and OCEAN faces are left valid, since those
# are genuine (Dirichlet) drainage boundaries where a real flux is physically
# meaningful. Outer boundary faces (ix==1/end for x, iy==1/end for y) are left
# at 1.0: compute_dhdx! etc. never write those entries anyway (their update
# ranges are 2:end-1), so they stay at their initialized value of zero.

# A face between a GROUNDED cell and a LAND/OCEAN cell whose fixed head lies above the grounded cell's
# carries water into the ice from the boundary. A fixed head is an unlimited reservoir, so a margin cell
# below higher ice-free land (head = bed elevation) would draw an unlimited supply that does not exist.
# With `land` (`ocean`) such LAND (OCEAN) faces are closed: the boundary takes water from the ice but
# does not feed it. OCEAN faces can stay two-way where ocean water may enter the bed (tidal intrusion).
@inline closes_inflow(m, land, ocean) = (land && m == LAND) || (ocean && m == OCEAN)
@inline inflow_face(m1, m2, h1, h2, land, ocean) = (m1 == GROUNDED && closes_inflow(m2, land, ocean) && h2 > h1) ||
                                                   (m2 == GROUNDED && closes_inflow(m1, land, ocean) && h1 > h2)

@parallel_indices (ix, iy) function compute_valid_x_kernel!(valid_x, mask, h, land, ocean)
    if ix <= size(valid_x, 1) && iy <= size(valid_x, 2)
        if ix > 1 && ix < size(valid_x, 1)
            m1, m2 = mask[ix-1, iy], mask[ix, iy]
            valid = (m1 != OTHER_BASIN) && (m1 != FROZEN_BED) && (m2 != OTHER_BASIN) && (m2 != FROZEN_BED) && # neither cell touching the face may be OTHER_BASIN or FROZEN_BED
                    !inflow_face(m1, m2, h[ix-1, iy], h[ix, iy], land, ocean)
            valid_x[ix, iy] = valid ? one(eltype(valid_x)) : zero(eltype(valid_x))
        else
            valid_x[ix, iy] = one(eltype(valid_x))
        end
    end
    return
end

@parallel_indices (ix, iy) function compute_valid_y_kernel!(valid_y, mask, h, land, ocean)
    if ix <= size(valid_y, 1) && iy <= size(valid_y, 2)
        if iy > 1 && iy < size(valid_y, 2)
            m1, m2 = mask[ix, iy-1], mask[ix, iy]
            valid = (m1 != OTHER_BASIN) && (m1 != FROZEN_BED) && (m2 != OTHER_BASIN) && (m2 != FROZEN_BED) && # neither cell touching the face may be OTHER_BASIN or FROZEN_BED
                    !inflow_face(m1, m2, h[ix, iy-1], h[ix, iy], land, ocean)
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

A face is invalid iff either cell it connects is `OTHER_BASIN` or `FROZEN_BED`: that cell's
hydrology isn't solved, so any gradient computed across that face would spuriously reflect a
frozen, non-evolving neighbour value rather than a real head/pressure difference. `LAND` and
`OCEAN` faces are left valid, since those are genuine (Dirichlet) drainage boundaries where a real flux
is physically meaningful.

With `land = true` (`ocean = true`), a `LAND` (`OCEAN`) face is additionally closed where the boundary's
fixed head lies above the neighbouring grounded cell's, so the boundary drains the ice but never feeds
it. Both are on by default (`ModelParameters` `outflow_only_land`/`outflow_only_ocean`); set
`outflow_only_ocean = false` to let ocean water into the bed (e.g. tidal intrusion).

The open/closed state is re-decided from the current head on every Picard iteration (see
`refresh_head_dependents!`). On the 8-dataset check this cost up to ~2x wall time (Thwaites 2 km,
pan-Antarctica 16 km). Deciding it once per time step instead was tested (branch
`outflow-faces-per-step`) and is not the fix: it was slower still (pan-Antarctica 16 km 150 s vs 55 s,
Helheim 500 m 23 s vs 9 s) and less robust (Helheim 500 m b_max 18.6 m vs 0.25 m), since faces lagged
by a step let boundary water in. The extra cost is the harder problem with those faces closed.
"""
function compute_face_masks!(s::State, land::Bool = false, ocean::Bool = false)
    @parallel compute_valid_x_kernel!(s.valid_x, s.mask, s.h, land, ocean)
    @parallel compute_valid_y_kernel!(s.valid_y, s.mask, s.h, land, ocean)
    return s
end
