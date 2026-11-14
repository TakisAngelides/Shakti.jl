# Cubic (Poiseuille) flux law through a sheet of gap height b, with an omega*Re term giving a
# Darcy-Weisbach-style correction towards turbulent (as opposed to laminar) flow resistance as Re
# grows: q = -K*grad(h), K = b^3*g / (12*nu*(1 + omega*Re)), Re = |q|/nu. compute_face_flux! below
# solves it on every face, lag-free: substituting Re = |q|/nu gives a quadratic in Re (Eq. 5,
# Sommers et al. 2018). With D = b^3*g*|dh| / (12*nu^2),
#   Re = D / (1 + omega*Re)  =>  omega*Re^2 + Re - D = 0
#   Re = (-1 + sqrt(1 + 4*omega*D)) / (2*omega)
# rearranged (multiply by the conjugate) to avoid catastrophic cancellation as omega -> 0
# (exactly our regime: omega = 1e-4 to 1e-3):
#   Re = 2*D / (1 + sqrt(1 + 4*omega*D))
# A lagged pair (q from the previous iteration's Re, then Re from q) can lock into a period-2
# oscillation for large gradients instead of converging (see git history on the
# exact-q-re-solve branch); it was removed with the face-mean-of-b path it used.

@parallel_indices (ix, iy) function compute_Re_kernel!(Re, Re_x, Re_y)
    if ix <= size(Re, 1) && iy <= size(Re, 2)
        Re[ix, iy] = (Re_x[ix, iy] + Re_x[ix+1, iy] + Re_y[ix, iy] + Re_y[ix, iy+1]) / 4
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.Re` (cell-centered Reynolds number) by averaging the four surrounding face values
(`s.Re_x`/`s.Re_y`).
"""
compute_Re!(s::State) = (@parallel compute_Re_kernel!(s.Re, s.Re_x, s.Re_y); s)

# Cell-centred K = b^3*g / (12*nu*(1 + omega*Re)), a diagnostic: the linear system assembles the face values
# K_x/K_y written by compute_face_flux!. It leaves out the free-surface factors (p.fs_factor, p.fs_omega).
@parallel_indices (ix, iy) function compute_K_kernel!(K, b, Re, ggrav, nu, omega)
    if ix <= size(K, 1) && iy <= size(K, 2)
        K[ix, iy] = (b[ix, iy]^3 * ggrav) / (12 * nu * (1 + omega * Re[ix, iy]))
    end
    return
end
compute_cell_K!(s::State, p::ModelParameters) = (@parallel compute_K_kernel!(s.K, s.b, s.Re, p.g, p.nu, p.omega); s)

# =============================================================================
# Face flux: one transmissivity for the solve AND the diagnostics
# =============================================================================
# The linear system and every flux-derived diagnostic (q, Re, the dissipation melt, D) must see
# the same face flux, or mass is conserved by one flux while melt/Re are computed from another.
# Per face: laminar transmissivity K0 = face_conductance(b^3*g/(12*nu)) of the two cells (the
# grounded cell's own value at an OCEAN/LAND face, 0 at a face touching OTHER_BASIN/FROZEN_BED or
# the domain edge); Re from the lag-free quadratic at the top of this file, with
# D = K0*|dh|/nu; then K_face = K0/(1 + omega*Re) and q = -K_face*dh (so |q|/nu == Re exactly).
# dsel_x/dsel_y: the head gradients dhdx/dhdy, whose sign picks the donor of Upwind/ImplicitUpwind (and whose size the
# blend of ImplicitUpwind).
# Free-surface layer (p.fs_factor, p.fs_omega, p.fs_blend). An unfilled cavity holds water on the bed with air above it:
# no shear at the surface, so the layer of depth W flows as the lower half of a duct of gap 2W (the mid-plane of a duct
# has no shear either). Laminar, that is q = g W^3 |grad h|/(3 nu), 4 times the duct's b^3 g/(12 nu) at the same depth;
# and the Reynolds number of the mirrored duct is 2|q|/nu, so omega counts twice (fs_omega = 2) for the same wall
# friction (Darcy-Weisbach f = 48 omega). Each cell carries a weight w from 0 (duct: filled, uplifted, or not GROUNDED)
# to 1 (free surface): its laminar transmissivity is c0 (1 + (fs_factor - 1) w) W^3, its omega omega (1 + (fs_omega - 1) w).
# w ramps up with the empty depth e = zb - h of an unfilled cell (s.stor > 0) over e = fs_blend * b (smoothstep; a jump
# at h = zb for fs_blend = 0). Re_x/Re_y stay |q|/nu.
@inline function free_surface_weight(stor, mask, h, zb, b, blend)
    e = zb - h
    (stor > 0 && mask == GROUNDED && e > 0) || return (zero(e), zero(e))
    w = blend * b
    (w > 0 && e < w) || return (one(e), zero(e))
    t = e / w
    return (t * t * (3 - 2 * t), -6 * t * (1 - t) / w) # the weight, and its slope in the head (de/dh = -1)
end
@parallel_indices (ix, iy) function free_surface_weight_kernel!(fsw, dfsw_dh, stor, mask, h, zb, b, blend)
    if ix <= size(fsw, 1) && iy <= size(fsw, 2)
        fsw[ix, iy], dfsw_dh[ix, iy] = free_surface_weight(stor[ix, iy], mask[ix, iy], h[ix, iy], zb[ix, iy], b[ix, iy], blend)
    end
    return
end
free_surface_on(p::ModelParameters) = p.fs_factor != 1 || p.fs_omega != 1
# The smallest fs_blend at which a cell's transmissivity never falls as its head rises. Inside the ramp the water thickness is
# b (1 - fs_blend t) and w = t^2 (3 - 2t), t = e/(fs_blend b); with a = fs_factor - 1, o = fs_omega - 1 the flux does not
# fall if 3 >= (1 - fs_blend t) 6t(1 - t)/fs_blend * c(t), with c = a/(1 + a w) laminar and a/(1 + a w) - o/(1 + o w)
# fully turbulent (in between, a mix of the two). 0.645 for fs_factor = 4, fs_omega = 2. Below it the equations can
# have two steady states per cell, and which one a run reaches depends on dt.
function fs_blend_monotone(fs_factor, fs_omega)
    a, o = Float64(fs_factor) - 1, Float64(fs_omega) - 1
    c(t) = (w = t^2 * (3 - 2t); max(a / (1 + a * w), a / (1 + a * w) - o / (1 + o * w)))
    ok(beta) = all(t -> (1 - beta * t) * 6t * (1 - t) / beta * c(t) <= 3, range(0, min(1, 1 / beta); length = 2001))
    ok(1e-12) && return 0.0
    lo, hi = 0.0, 1.0
    while !ok(hi)
        lo, hi = hi, 2hi
    end
    for _ in 1:50
        mid = (lo + hi) / 2
        ok(mid) ? (hi = mid) : (lo = mid)
    end
    return hi
end
compute_free_surface_weight!(s::State, p::ModelParameters) =
    (free_surface_on(p) && @parallel free_surface_weight_kernel!(s.fsw, s.dfsw_dh, s.stor, s.mask, s.h, s.zb, s.b, p.fs_blend); s)
# The weight of a face: of the cell whose transmissivity the face takes (the donor under Upwind, the grounded cell at an
# OCEAN/LAND face), the mean of the two for a mean of the two transmissivities. Under ImplicitUpwind it blends between
# two grounded cells as the transmissivity does (face_mean): w_thin + (w_up - w_thin) r, so the face's omega follows the
# same mix of the two cells as its K0, and it is the donor's from |grad h| = grad_blend on.
@inline face_weight(K0, Ka, Kb, wa, wb) = K0 == Ka ? wa : (K0 == Kb ? wb : (wa + wb) / 2)
@inline face_weight(::AbstractKFaceScheme, K0, Ka, ma, Kb, mb, wa, wb, dh) = face_weight(K0, Ka, Kb, wa, wb)
@inline function face_weight(kfs::ImplicitUpwind, K0, Ka, ma, Kb, mb, wa, wb, dh)
    (ma == GROUNDED && mb == GROUNDED) || return face_weight(K0, Ka, Kb, wa, wb)
    wlo = Ka <= Kb ? wa : wb # the thinner cell's, as min(Ka, Kb) in face_mean
    wup = dh > 0 ? wb : (dh < 0 ? wa : wlo)
    return wlo + (wup - wlo) * upwind_share(kfs, dh)
end

@parallel_indices (ix, iy) function compute_face_flux_kernel!(K_x, K_y, q_x, q_y, Re_x, Re_y, b, mask, dhdx, dhdy, valid_x, valid_y, c0, nu, omega, kfs::AbstractKFaceScheme, dsel_x, dsel_y, fsw, fs_c, fs_o)
    nx, ny = size(b, 1), size(b, 2)
    if ix <= size(K_x, 1) && iy <= size(K_x, 2)
        K0, om = zero(c0), omega
        if ix > 1 && ix <= nx
            Ka, Kb = c0 * (1 + fs_c * fsw[ix-1, iy]) * b[ix-1, iy]^3, c0 * (1 + fs_c * fsw[ix, iy]) * b[ix, iy]^3
            K0 = face_conductance(kfs, Ka, mask[ix-1, iy], Kb, mask[ix, iy], dsel_x[ix, iy])
            om = omega * (1 + fs_o * face_weight(kfs, K0, Ka, mask[ix-1, iy], Kb, mask[ix, iy], fsw[ix-1, iy], fsw[ix, iy], dsel_x[ix, iy]))
            K0 *= valid_x[ix, iy]
        end
        D = K0 * abs(dhdx[ix, iy]) / nu
        Re = 2 * D / (1 + sqrt(1 + 4 * om * D))
        Kf = K0 / (1 + om * Re)
        K_x[ix, iy] = Kf
        Re_x[ix, iy] = Re
        q_x[ix, iy] = -Kf * dhdx[ix, iy]
    end
    if ix <= size(K_y, 1) && iy <= size(K_y, 2)
        K0, om = zero(c0), omega
        if iy > 1 && iy <= ny
            Ka, Kb = c0 * (1 + fs_c * fsw[ix, iy-1]) * b[ix, iy-1]^3, c0 * (1 + fs_c * fsw[ix, iy]) * b[ix, iy]^3
            K0 = face_conductance(kfs, Ka, mask[ix, iy-1], Kb, mask[ix, iy], dsel_y[ix, iy])
            om = omega * (1 + fs_o * face_weight(kfs, K0, Ka, mask[ix, iy-1], Kb, mask[ix, iy], fsw[ix, iy-1], fsw[ix, iy], dsel_y[ix, iy]))
            K0 *= valid_y[ix, iy]
        end
        D = K0 * abs(dhdy[ix, iy]) / nu
        Re = 2 * D / (1 + sqrt(1 + 4 * om * D))
        Kf = K0 / (1 + om * Re)
        K_y[ix, iy] = Kf
        Re_y[ix, iy] = Re
        q_y[ix, iy] = -Kf * dhdy[ix, iy]
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates the face transmissivities `s.K_x`/`s.K_y`, fluxes `s.q_x`/`s.q_y` and Reynolds numbers
`s.Re_x`/`s.Re_y` together, lag-free, from the current `s.b_w` (the water thickness, `s.b` unless cavities are
unfilled, see [`compute_b_w!`](@ref))/`s.dhdx`/`s.dhdy` -- the version
used in the Picard loop ([`refresh_head_dependents!`](@ref)).

# Notes

`s.K_x`/`s.K_y` are exactly the coefficients the linear system assembles (`linear_solver.jl`), so
the flux the solve conserves and the `q` every diagnostic reads (Re, the dissipation melt term,
SUHMO's `D`, output) are the same numbers. Face rules come from [`face_conductance`](@ref): at an
`OCEAN`/`LAND` face the grounded cell's own gap height sets the transmissivity (averaging in the
Dirichlet cell's placeholder `b = 0` would cut the outlet flux by a factor of 8). With
`D = K0*|dh|/nu` (`K0` the laminar face transmissivity), `omega*Re^2 + Re - D = 0` is solved as
`Re = 2*D / (1 + sqrt(1 + 4*omega*D))` (see the notes at the top of `water_flux.jl`). With `p.fs_factor`/`p.fs_omega`
set, an unfilled cell's transmissivity is `fs_factor` times larger and its `omega` `fs_omega` times larger, weighted by
`s.fsw` (`free_surface_weight`); `s.Re_x`/`s.Re_y` stay `|q|/nu`.
"""
compute_face_flux!(s::State, p::ModelParameters, kfs::AbstractKFaceScheme = Arithmetic(), gc::AbstractGapCoupling = ExplicitGapCoupling()) =
    (compute_b_w!(s, p); compute_free_surface_weight!(s, p); @parallel compute_face_flux_kernel!(s.K_x, s.K_y, s.q_x, s.q_y, s.Re_x, s.Re_y, s.b_w, s.mask, s.dhdx, s.dhdy, s.valid_x, s.valid_y, p.g / (12 * p.nu), p.nu, p.omega, kfs, s.dhdx, s.dhdy, s.fsw, p.fs_factor - 1, p.fs_omega - 1); s)

# Under ImplicitUpwind the face flux q = -K(W_d)*dphi of a face depends on the head through the water thickness W_d of
# its donor d (the higher-potential cell), where that depth is free (s.dbw_dh = 1: unfilled or uplifted). With
# K0 = c0*W^3 and Re from omega*Re^2 + Re - K0*|dphi|/nu = 0 (compute_face_flux_kernel!), the derivative of
# |q| = nu*Re at fixed dphi is d|q|/dW = 3*K0*|dphi|/(W*(1 + 2*omega*Re)) (3 K/W laminar, 3/2 K/W turbulent) where the
# face takes the donor's K0 whole (no blend, no free-surface weight; the general form is below), and
# per head of the donor J = d|q|/dW * dW/dh [m/s] (the head row divides it by the face spacing). Jw_x[i] stores +J if the
# donor is the cell at the high index i and -J if it is the cell at i - 1; 0 if the donor's depth is not free, if it
# is not GROUNDED (inflow from an OCEAN/LAND cell: the face takes the grounded cell's transmissivity, see
# face_conductance, which this head row does not linearize), or if the face carries no flux.
# The face blends (ImplicitUpwind): K0 = K_thin + (K_up - K_thin) r and w_f = w_thin + (w_up - w_thin) r (face_mean,
# face_weight), r = r(|dphi|/grad_blend). Where the donor is the thicker cell, K0 and w_f move with its head through its
# share r of them; where it is the thinner one (or the other cell is not GROUNDED) K0 = K_d and w_f = w_d move whole
# (share 1). The thinner cell's (1 - r) part on a face it does not feed is not linearized (its dF/dh has the opposite
# sign there, which would break the M-matrix pattern), nor is r's dependence on dphi: both are iterated as plain fixed
# points.
# With the free-surface weight w(h) of the donor (free_surface_weight), K_d = c0 (1 + a w) W^3 and the face's
# omega_f = omega (1 + o w_f) (a = fs_factor - 1, o = fs_omega - 1) move with the head too: per head of the donor, with
# its share s,
#   d|q|/dh = s [K_d |dphi| (3 dW/dh / W + a dw/dh / (1 + a w)) - nu Re^2 omega o dw/dh] / (1 + 2 omega_f Re),
# which is 3 K/W dW/dh (1 + omega Re)/(1 + 2 omega Re) for w = 0 and s = 1 (then K_d = K0 = K (1 + omega Re)).
# dw/dh <= 0 (filling turns the layer into a duct, 4 times less conductive), and where that outweighs the deeper water
# the flux falls as the head rises; J is then kept at 0 (a negative J breaks the M-matrix pattern).
@inline function free_water_newton(Kd, Re, dphi, W, dW, om_f, omega, w, dw, a, o, nu, s)
    J = s * (Kd * dphi * (3 * dW / W + a * dw / (1 + a * w)) - nu * Re^2 * omega * o * dw) / (1 + 2 * om_f * Re)
    return max(J, zero(J))
end
# The donor's share of a blended face between two GROUNDED cells (see above): r where it is the thicker cell, 1 where it
# is the thinner one.
@inline donor_share(kfs::ImplicitUpwind, g, Kd, Ko) = Kd <= Ko ? one(g) : upwind_share(kfs, g)
# Newton coefficient of one face (a at the low index, b at the high one; g = dphi across it, nonzero): +J for donor b,
# -J for donor a, 0 where the donor is not GROUNDED or its depth is not free.
@inline function face_newton(kfs, g, Kf, Re, Wa, Wb, dWa, dWb, ma, mb, wa, wb, dwa, dwb, c0, omega, fs_c, fs_o, nu)
    hi = g > 0 # the donor is b, the higher-potential cell
    Wd, dWd, md, wd, dwd = hi ? (Wb, dWb, mb, wb, dwb) : (Wa, dWa, ma, wa, dwa)
    (md == GROUNDED && dWd > 0) || return zero(Kf)
    Ka, Kb = c0 * (1 + fs_c * wa) * Wa^3, c0 * (1 + fs_c * wb) * Wb^3
    Kd, Ko, mo = hi ? (Kb, Ka, ma) : (Ka, Kb, mb)
    s = mo == GROUNDED ? donor_share(kfs, g, Kd, Ko) : one(g)
    om_f = omega * (1 + fs_o * face_weight(kfs, Kd, Ka, ma, Kb, mb, wa, wb, g))
    J = free_water_newton(Kd, Re, abs(g), Wd, dWd, om_f, omega, wd, dwd, fs_c, fs_o, nu, s)
    return hi ? J : -J
end
@parallel_indices (ix, iy) function water_depth_newton_kernel!(Jw_x, Jw_y, K_x, K_y, Re_x, Re_y, dhdx, dhdy, b_w, dbw_dh, mask, c0, omega, kfs, fsw, dfsw_dh, fs_c, fs_o, nu)
    nx, ny = size(b_w, 1), size(b_w, 2)
    if ix <= size(Jw_x, 1) && iy <= size(Jw_x, 2)
        J = zero(omega)
        if ix > 1 && ix <= nx && K_x[ix, iy] > 0 && dhdx[ix, iy] != 0
            J = face_newton(kfs, dhdx[ix, iy], K_x[ix, iy], Re_x[ix, iy], b_w[ix-1, iy], b_w[ix, iy], dbw_dh[ix-1, iy], dbw_dh[ix, iy],
                            mask[ix-1, iy], mask[ix, iy], fsw[ix-1, iy], fsw[ix, iy], dfsw_dh[ix-1, iy], dfsw_dh[ix, iy], c0, omega, fs_c, fs_o, nu)
        end
        Jw_x[ix, iy] = J
    end
    if ix <= size(Jw_y, 1) && iy <= size(Jw_y, 2)
        J = zero(omega)
        if iy > 1 && iy <= ny && K_y[ix, iy] > 0 && dhdy[ix, iy] != 0
            J = face_newton(kfs, dhdy[ix, iy], K_y[ix, iy], Re_y[ix, iy], b_w[ix, iy-1], b_w[ix, iy], dbw_dh[ix, iy-1], dbw_dh[ix, iy],
                            mask[ix, iy-1], mask[ix, iy], fsw[ix, iy-1], fsw[ix, iy], dfsw_dh[ix, iy-1], dfsw_dh[ix, iy], c0, omega, fs_c, fs_o, nu)
        end
        Jw_y[ix, iy] = J
    end
    return
end

"""
$(TYPEDSIGNATURES)

[`ImplicitUpwind`](@ref) method: the face fluxes from the donor's water thickness at the end of the step
([`compute_b_w_iterate!`](@ref), from the current head iterate and the gap coupling `gc`), blended towards the thinner
cell's where the head gradient is below `kfs.grad_blend`, and the Newton coefficients `s.Jw_x`/`s.Jw_y` of
their dependence on the donor's head through that depth (`water_depth_newton_kernel!`), which the head
rows add (`add_water_depth_newton!`, `linear_solver.jl`).

# Notes

Until 2026-11 the donor jumped with the sign of the head difference, and the donors were tracked through the step
instead: a face whose donor had flipped four times kept it, and once the head had converged a face held on its
downstream side was moved upstream, then to its thinner cell, and the head iterated again (up to three rounds). On
Helheim (500 m, dt = 30 min, from day 270 of a year) that took 32 iterations a step, 17 of them after the head had first
converged; the blend at `grad_blend = 1e-4` takes about 13 (see `ImplicitUpwind`).
"""
function compute_face_flux!(s::State, p::ModelParameters, kfs::ImplicitUpwind, gc::AbstractGapCoupling = ExplicitGapCoupling())
    compute_b_w_iterate!(s, p, gc)
    compute_free_surface_weight!(s, p)
    @parallel compute_face_flux_kernel!(s.K_x, s.K_y, s.q_x, s.q_y, s.Re_x, s.Re_y, s.b_w, s.mask, s.dhdx, s.dhdy, s.valid_x, s.valid_y, p.g / (12 * p.nu), p.nu, p.omega, kfs, s.dhdx, s.dhdy, s.fsw, p.fs_factor - 1, p.fs_omega - 1)
    @parallel water_depth_newton_kernel!(s.Jw_x, s.Jw_y, s.K_x, s.K_y, s.Re_x, s.Re_y, s.dhdx, s.dhdy, s.b_w, s.dbw_dh, s.mask, p.g / (12 * p.nu), p.omega, kfs, s.fsw, s.dfsw_dh, p.fs_factor - 1, p.fs_omega - 1, p.nu)
    return s
end

"""
$(TYPEDSIGNATURES)

Zeroes the previous head steps of the oscillation damping (`s.dh_it`, [`limit_head_step!`](@ref)) at the start of a
step, for every K-face scheme.
"""
reset_donors!(::AbstractKFaceScheme, s::State) = (fill!(s.dh_it, 0); s)
