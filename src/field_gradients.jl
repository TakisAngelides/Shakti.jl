# =============================================================================
# Per-Picard-iteration fields
# =============================================================================
# Everything below is recomputed every Picard iteration, driven by the head h
# most recently produced by the linear solve (refresh_head_dependents! in
# elliptic_solver.jl calls the fused compute_dhdxy!/compute_dpwdxy!; the
# single-direction kernels are used by set_initial_conditions!).

# All of them zero the result on any face flagged invalid by compute_face_masks!
# (a face touching an OTHER_BASIN or FROZEN_BED cell, or a LAND/OCEAN face closed
# as outflow-only), so downstream flux/melt computations never see a spurious
# gradient driven by a frozen, non-evolving neighbour value. First/last row (x) or column (y) are left
# as zero: undefined at the grid domain boundary. dhdx/dhdy feed compute_face_flux!
# (water_flux.jl); dpwdx/dpwdy (the gradient of the water pressure at the bed) feed the "sensible" heat term in
# compute_mdot! further down.

# Head difference across a face plus the gap jump `db` (nonzero only under SheetTopPotential, see
# hydraulic_potential.jl); skipping the addition when `db` is zero keeps BedPotential bit-for-bit.
@inline face_head_difference(dh, db) = iszero(db) ? dh : dh + db

@parallel_indices (ix, iy) function compute_dhdx_kernel!(dhdx, h, db_x, valid_x, dx)
    if ix > 1 && ix < size(dhdx, 1) && iy <= size(dhdx, 2)
        dhdx[ix, iy] = (face_head_difference(h[ix, iy] - h[ix-1, iy], db_x[ix, iy]) / dx) * valid_x[ix, iy] # staggered derivative from two center values times the validation of the face which gets zeroed if touching OTHER_BASIN
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.dhdx` (x-gradient of the head that drives the flux) from the current `s.h`, plus the gap
jump `s.db_x` under [`SheetTopPotential`](@ref) (zero otherwise), zeroed on any face touching an
`OTHER_BASIN` cell (`s.valid_x`) so downstream flux/melt computations never see a spurious gradient
driven by a frozen, non-evolving value. Feeds [`compute_face_flux!`](@ref).
"""
function compute_dhdx!(s::State, g::Grid)
    @parallel compute_dhdx_kernel!(s.dhdx, s.h, s.db_x, s.valid_x, g.dx)
    return s
end

@parallel_indices (ix, iy) function compute_dhdy_kernel!(dhdy, h, db_y, valid_y, dy)
    if iy > 1 && iy < size(dhdy, 2) && ix <= size(dhdy, 1)
        dhdy[ix, iy] = (face_head_difference(h[ix, iy] - h[ix, iy-1], db_y[ix, iy]) / dy) * valid_y[ix, iy]
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.dhdy` (y-gradient of hydraulic head), the y-face counterpart of
[`compute_dhdx!`](@ref).
"""
function compute_dhdy!(s::State, g::Grid)
    @parallel compute_dhdy_kernel!(s.dhdy, s.h, s.db_y, s.valid_y, g.dy)
    return s
end

# The pressure of the sensible-heat term (Q_sens = ct cw rho_w q . grad(p), compute_mdot!) is the water pressure at the
# bed, p_b = rho_w g (hhat - zb), with hhat the head that drives the flux (h, plus the gap b under SheetTopPotential). It
# is not floored or capped: p_b = rho_w g W in an unfilled cell, pw + rho_w g b in a filled one, po + rho_w g W in an
# uplifted one. The pressure pw at the top of the sheet (compute_pw!) differs from it by rho_w g W, and its gradient
# would add -rho_w g grad(W) to grad(p_b): in a filled cell a gradient of the start-of-step gap, an explicit centred
# advection of b by the flux that lets a cell-to-cell sawtooth of the gap grow (Felden et al. 2023 Fig. 4 at dx = 0.125 m).
# Bed pressure of a cell (from the head h, without the gap), and its difference across a face plus the face's gap jump
# (zero under BedPotential: skipping the addition keeps BedPotential bit for bit, and under FilledCavities and NoUplift
# the result equals the gradient of s.pw bit for bit).
@inline bed_pressure(h, zb, rho_w, ggrav) = rho_w * ggrav * (h - zb)
@inline face_bed_pressure_difference(pP, pF, db, rho_w, ggrav) = iszero(db) ? pP - pF : (pP - pF) + rho_w * ggrav * db

@parallel_indices (ix, iy) function compute_dpwdx_kernel!(dpwdx, h, zb, db_x, valid_x, dx, rho_w, ggrav)
    if ix > 1 && ix < size(dpwdx, 1) && iy <= size(dpwdx, 2)
        dp = face_bed_pressure_difference(bed_pressure(h[ix, iy], zb[ix, iy], rho_w, ggrav), bed_pressure(h[ix-1, iy], zb[ix-1, iy], rho_w, ggrav), db_x[ix, iy], rho_w, ggrav)
        dpwdx[ix, iy] = (dp / dx) * valid_x[ix, iy]
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.dpwdx`, the x-gradient of the water pressure at the bed `p_b = rho_w g (hhat - zb)` (not floored or capped,
see the notes above), from the current `s.h`, `s.zb` and the gap jump `s.db_x` under [`SheetTopPotential`](@ref), zeroed
the same way as [`compute_dhdx!`](@ref). Feeds the sensible-heat term in `compute_mdot!` (`melt_rate.jl`). Under
[`FilledCavities`](@ref), [`NoUplift`](@ref) and [`BedPotential`](@ref) this is the gradient of `s.pw`, bit for bit.
"""
function compute_dpwdx!(s::State, g::Grid, p::ModelParameters)
    @parallel compute_dpwdx_kernel!(s.dpwdx, s.h, s.zb, s.db_x, s.valid_x, g.dx, p.rho_w, p.g)
    return s
end

@parallel_indices (ix, iy) function compute_dpwdy_kernel!(dpwdy, h, zb, db_y, valid_y, dy, rho_w, ggrav)
    if iy > 1 && iy < size(dpwdy, 2) && ix <= size(dpwdy, 1)
        dp = face_bed_pressure_difference(bed_pressure(h[ix, iy], zb[ix, iy], rho_w, ggrav), bed_pressure(h[ix, iy-1], zb[ix, iy-1], rho_w, ggrav), db_y[ix, iy], rho_w, ggrav)
        dpwdy[ix, iy] = (dp / dy) * valid_y[ix, iy]
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.dpwdy`, the y-gradient of the water pressure at the bed, the y-face counterpart of
[`compute_dpwdx!`](@ref).
"""
function compute_dpwdy!(s::State, g::Grid, p::ModelParameters)
    @parallel compute_dpwdy_kernel!(s.dpwdy, s.h, s.zb, s.db_y, s.valid_y, g.dy, p.rho_w, p.g)
    return s
end

# Fused hot-path versions: one @parallel launch instead of two. ParallelStencil
# infers the launch range from the elementwise max of every array argument's
# size (see ParallelKernel.maxsize), so passing both the (nx+1,ny) x-face and
# (nx,ny+1) y-face arrays as arguments launches over their union bounding box
# (1:nx+1, 1:ny+1) automatically -- each `if` below still only writes within
# its own field's bounds, exactly as the unfused kernels above already did.
@parallel_indices (ix, iy) function compute_dhdxy_kernel!(dhdx, dhdy, h, db_x, db_y, valid_x, valid_y, dx, dy)
    if ix > 1 && ix < size(dhdx, 1) && iy <= size(dhdx, 2)
        dhdx[ix, iy] = (face_head_difference(h[ix, iy] - h[ix-1, iy], db_x[ix, iy]) / dx) * valid_x[ix, iy] # valid_x is an XFace field; 0 if face touches a 'OTHER_BASIN' cell or 1 otherwise
    end
    if iy > 1 && iy < size(dhdy, 2) && ix <= size(dhdy, 1)
        dhdy[ix, iy] = (face_head_difference(h[ix, iy] - h[ix, iy-1], db_y[ix, iy]) / dy) * valid_y[ix, iy]
    end
    return
end
"""
$(TYPEDSIGNATURES)

Fused version of [`compute_dhdx!`](@ref) + [`compute_dhdy!`](@ref): one `@parallel` launch
instead of two.
"""
compute_dhdxy!(s::State, g::Grid) = (@parallel compute_dhdxy_kernel!(s.dhdx, s.dhdy, s.h, s.db_x, s.db_y, s.valid_x, s.valid_y, g.dx, g.dy); s)

@parallel_indices (ix, iy) function compute_dpwdxy_kernel!(dpwdx, dpwdy, h, zb, db_x, db_y, valid_x, valid_y, dx, dy, rho_w, ggrav)
    if ix > 1 && ix < size(dpwdx, 1) && iy <= size(dpwdx, 2)
        dp = face_bed_pressure_difference(bed_pressure(h[ix, iy], zb[ix, iy], rho_w, ggrav), bed_pressure(h[ix-1, iy], zb[ix-1, iy], rho_w, ggrav), db_x[ix, iy], rho_w, ggrav)
        dpwdx[ix, iy] = (dp / dx) * valid_x[ix, iy]
    end
    if iy > 1 && iy < size(dpwdy, 2) && ix <= size(dpwdy, 1)
        dp = face_bed_pressure_difference(bed_pressure(h[ix, iy], zb[ix, iy], rho_w, ggrav), bed_pressure(h[ix, iy-1], zb[ix, iy-1], rho_w, ggrav), db_y[ix, iy], rho_w, ggrav)
        dpwdy[ix, iy] = (dp / dy) * valid_y[ix, iy]
    end
    return
end
"""
$(TYPEDSIGNATURES)

Fused version of [`compute_dpwdx!`](@ref) + [`compute_dpwdy!`](@ref): one `@parallel` launch
instead of two.
"""
compute_dpwdxy!(s::State, g::Grid, p::ModelParameters) = (@parallel compute_dpwdxy_kernel!(s.dpwdx, s.dpwdy, s.h, s.zb, s.db_x, s.db_y, s.valid_x, s.valid_y, g.dx, g.dy, p.rho_w, p.g); s)