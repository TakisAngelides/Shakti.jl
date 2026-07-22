# =============================================================================
# Per-Picard-iteration fields
# =============================================================================
# Everything below is recomputed every Picard iteration, driven by the head h
# most recently produced by the linear solve (see Picard_iteration! in
# elliptic_solver.jl, which calls these in the same order they appear here).

# All four zero the result on any face flagged invalid by compute_face_masks!
# (i.e. any face touching an OTHER_BASIN cell), so downstream flux/melt
# computations never see a spurious gradient driven by a frozen,
# non-evolving OTHER_BASIN value. First/last row (x) or column (y) are left
# as zero: undefined at the grid domain boundary. dhdx/dhdy feed compute_q_x!/
# compute_q_y! below; dpwdx/dpwdy feed the "sensible" heat term in
# compute_mdot! further down.

@parallel_indices (ix, iy) function compute_dhdx_kernel!(dhdx, h, valid_x, dx)
    if ix > 1 && ix < size(dhdx, 1) && iy <= size(dhdx, 2)
        dhdx[ix, iy] = ((h[ix, iy] - h[ix-1, iy]) / dx) * valid_x[ix, iy]
    end
    return
end

function compute_dhdx!(s::State, g::Grid)
    @parallel compute_dhdx_kernel!(s.dhdx, s.h, s.valid_x, g.dx)
    return s
end

@parallel_indices (ix, iy) function compute_dhdy_kernel!(dhdy, h, valid_y, dy)
    if iy > 1 && iy < size(dhdy, 2) && ix <= size(dhdy, 1)
        dhdy[ix, iy] = ((h[ix, iy] - h[ix, iy-1]) / dy) * valid_y[ix, iy]
    end
    return
end

function compute_dhdy!(s::State, g::Grid)
    @parallel compute_dhdy_kernel!(s.dhdy, s.h, s.valid_y, g.dy)
    return s
end

@parallel_indices (ix, iy) function compute_dpwdx_kernel!(dpwdx, pw, valid_x, dx)
    if ix > 1 && ix < size(dpwdx, 1) && iy <= size(dpwdx, 2)
        dpwdx[ix, iy] = ((pw[ix, iy] - pw[ix-1, iy]) / dx) * valid_x[ix, iy]
    end
    return
end

function compute_dpwdx!(s::State, g::Grid)
    @parallel compute_dpwdx_kernel!(s.dpwdx, s.pw, s.valid_x, g.dx)
    return s
end

@parallel_indices (ix, iy) function compute_dpwdy_kernel!(dpwdy, pw, valid_y, dy)
    if iy > 1 && iy < size(dpwdy, 2) && ix <= size(dpwdy, 1)
        dpwdy[ix, iy] = ((pw[ix, iy] - pw[ix, iy-1]) / dy) * valid_y[ix, iy]
    end
    return
end

function compute_dpwdy!(s::State, g::Grid)
    @parallel compute_dpwdy_kernel!(s.dpwdy, s.pw, s.valid_y, g.dy)
    return s
end

# Fused hot-path versions: one @parallel launch instead of two. ParallelStencil
# infers the launch range from the elementwise max of every array argument's
# size (see ParallelKernel.maxsize), so passing both the (nx+1,ny) x-face and
# (nx,ny+1) y-face arrays as arguments launches over their union bounding box
# (1:nx+1, 1:ny+1) automatically -- each `if` below still only writes within
# its own field's bounds, exactly as the unfused kernels above already did.
@parallel_indices (ix, iy) function compute_dhdxy_kernel!(dhdx, dhdy, h, valid_x, valid_y, dx, dy)
    if ix > 1 && ix < size(dhdx, 1) && iy <= size(dhdx, 2)
        dhdx[ix, iy] = ((h[ix, iy] - h[ix-1, iy]) / dx) * valid_x[ix, iy]
    end
    if iy > 1 && iy < size(dhdy, 2) && ix <= size(dhdy, 1)
        dhdy[ix, iy] = ((h[ix, iy] - h[ix, iy-1]) / dy) * valid_y[ix, iy]
    end
    return
end
compute_dhdxy!(s::State, g::Grid) = (@parallel compute_dhdxy_kernel!(s.dhdx, s.dhdy, s.h, s.valid_x, s.valid_y, g.dx, g.dy); s)

@parallel_indices (ix, iy) function compute_dpwdxy_kernel!(dpwdx, dpwdy, pw, valid_x, valid_y, dx, dy)
    if ix > 1 && ix < size(dpwdx, 1) && iy <= size(dpwdx, 2)
        dpwdx[ix, iy] = ((pw[ix, iy] - pw[ix-1, iy]) / dx) * valid_x[ix, iy]
    end
    if iy > 1 && iy < size(dpwdy, 2) && ix <= size(dpwdy, 1)
        dpwdy[ix, iy] = ((pw[ix, iy] - pw[ix, iy-1]) / dy) * valid_y[ix, iy]
    end
    return
end
compute_dpwdxy!(s::State, g::Grid) = (@parallel compute_dpwdxy_kernel!(s.dpwdx, s.dpwdy, s.pw, s.valid_x, s.valid_y, g.dx, g.dy); s)