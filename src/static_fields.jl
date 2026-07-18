# =============================================================================
# Static / initial geometry fields
# =============================================================================
# Computed once (e.g. by initial-condition setup) rather than every Picard
# iteration: H/po/h derive from the *initial* b, zb, zs, pw and don't change
# until a full time step has evolved b (see the gap-height section below).

@parallel_indices (ix, iy) function compute_H_kernel!(H, zs, zb, b)
    if ix <= size(H, 1) && iy <= size(H, 2)
        H[ix, iy] = zs[ix, iy] - (zb[ix, iy] + b[ix, iy])
    end
    return
end
compute_H!(s::State) = (@parallel compute_H_kernel!(s.H, s.zs, s.zb, s.b); s)

@parallel_indices (ix, iy) function compute_po_kernel!(po, H, rho_i, ggrav)
    if ix <= size(po, 1) && iy <= size(po, 2)
        po[ix, iy] = rho_i * ggrav * H[ix, iy]
    end
    return
end
compute_po!(s::State, p::ModelParameters) = (@parallel compute_po_kernel!(s.po, s.H, p.rho_i, p.g); s)

# Inverse of compute_pw! below: seeds h from a prescribed initial pw. Once the
# time loop is running, h is instead the Picard solver's primary unknown, and
# pw is derived FROM h every iteration (compute_pw!), not the other way round.
@parallel_indices (ix, iy) function compute_h_kernel!(h, pw, zb, rho_w, ggrav)
    if ix <= size(h, 1) && iy <= size(h, 2)
        h[ix, iy] = pw[ix, iy] / (rho_w * ggrav) + zb[ix, iy]
    end
    return
end
compute_h!(s::State, p::ModelParameters) = (@parallel compute_h_kernel!(s.h, s.pw, s.zb, p.rho_w, p.g); s)

@parallel_indices (ix, iy) function compute_abs_ub_kernel!(abs_ub, ub_x, ub_y)
    if ix <= size(abs_ub, 1) && iy <= size(abs_ub, 2)
        ubx_c = (ub_x[ix, iy] + ub_x[ix+1, iy]) / 2
        uby_c = (ub_y[ix, iy] + ub_y[ix, iy+1]) / 2
        abs_ub[ix, iy] = sqrt(ubx_c^2 + uby_c^2)
    end
    return
end
compute_abs_ub!(s::State) = (@parallel compute_abs_ub_kernel!(s.abs_ub, s.ub_x, s.ub_y); s)