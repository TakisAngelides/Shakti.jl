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
# Per face: laminar conductance K0 = face_conductance(b^3*g/(12*nu)) of the two cells (the
# grounded cell's own value at an OCEAN/LAND face, 0 at a face touching OTHER_BASIN/FROZEN_BED or
# the domain edge); Re from the lag-free quadratic at the top of this file, with
# D = K0*|dh|/nu; then K_face = K0/(1 + omega*Re) and q = -K_face*dh (so |q|/nu == Re exactly).
# dsel_x/dsel_y: the face values whose sign picks the donor of Upwind/ImplicitUpwind: dhdx/dhdy themselves, or the
# ImplicitUpwind donors s.don_x/s.don_y (update_donor_kernel!).
# Free-surface layer (p.fs_factor, p.fs_omega, p.fs_blend). An unfilled cavity holds water on the bed with air above it:
# no shear at the surface, so the layer of depth W flows as the lower half of a duct of gap 2W (the mid-plane of a duct
# has no shear either). Laminar, that is q = g W^3 |grad h|/(3 nu), 4 times the duct's b^3 g/(12 nu) at the same depth;
# and the Reynolds number of the mirrored duct is 2|q|/nu, so omega counts twice (fs_omega = 2) for the same wall
# friction (Darcy-Weisbach f = 48 omega). Each cell carries a weight w from 0 (duct: filled, uplifted, or not GROUNDED)
# to 1 (free surface): its laminar conductance is c0 (1 + (fs_factor - 1) w) W^3, its omega omega (1 + (fs_omega - 1) w).
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
compute_free_surface_weight!(s::State, p::ModelParameters) =
    (free_surface_on(p) && @parallel free_surface_weight_kernel!(s.fsw, s.dfsw_dh, s.stor, s.mask, s.h, s.zb, s.b, p.fs_blend); s)
# The weight of a face: of the cell whose conductance the face takes (the donor under Upwind/ImplicitUpwind, the grounded
# cell at an OCEAN/LAND face), the mean of the two for a mean of the two conductances.
@inline face_weight(K0, Ka, Kb, wa, wb) = K0 == Ka ? wa : (K0 == Kb ? wb : (wa + wb) / 2)

@parallel_indices (ix, iy) function compute_face_flux_kernel!(K_x, K_y, q_x, q_y, Re_x, Re_y, b, mask, dhdx, dhdy, valid_x, valid_y, c0, nu, omega, kfs::AbstractKFaceScheme, dsel_x, dsel_y, fsw, fs_c, fs_o)
    nx, ny = size(b, 1), size(b, 2)
    if ix <= size(K_x, 1) && iy <= size(K_x, 2)
        K0, om = zero(c0), omega
        if ix > 1 && ix <= nx
            Ka, Kb = c0 * (1 + fs_c * fsw[ix-1, iy]) * b[ix-1, iy]^3, c0 * (1 + fs_c * fsw[ix, iy]) * b[ix, iy]^3
            K0 = face_conductance(kfs, Ka, mask[ix-1, iy], Kb, mask[ix, iy], face_select(kfs, dsel_x[ix, iy]))
            om = omega * (1 + fs_o * face_weight(K0, Ka, Kb, fsw[ix-1, iy], fsw[ix, iy]))
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
            K0 = face_conductance(kfs, Ka, mask[ix, iy-1], Kb, mask[ix, iy], face_select(kfs, dsel_y[ix, iy]))
            om = omega * (1 + fs_o * face_weight(K0, Ka, Kb, fsw[ix, iy-1], fsw[ix, iy]))
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
`s.Re_x`/`s.Re_y` together, lag-free, from the current `s.b_w` (the water depth, `s.b` unless cavities are
unfilled, see [`compute_b_w!`](@ref))/`s.dhdx`/`s.dhdy` -- the version
used in the Picard loop ([`refresh_head_dependents!`](@ref)).

# Notes

`s.K_x`/`s.K_y` are exactly the coefficients the linear system assembles (`linear_solver.jl`), so
the flux the solve conserves and the `q` every diagnostic reads (Re, the dissipation melt term,
SUHMO's `D`, output) are the same numbers. Face rules come from [`face_conductance`](@ref): at an
`OCEAN`/`LAND` face the grounded cell's own gap height sets the conductance (averaging in the
Dirichlet cell's placeholder `b = 0` would cut the outlet flux by a factor of 8). With
`D = K0*|dh|/nu` (`K0` the laminar face conductance), `omega*Re^2 + Re - D = 0` is solved as
`Re = 2*D / (1 + sqrt(1 + 4*omega*D))` (see the notes at the top of `water_flux.jl`). With `p.fs_factor`/`p.fs_omega`
set, an unfilled cell's conductance is `fs_factor` times larger and its `omega` `fs_omega` times larger, weighted by
`s.fsw` (`free_surface_weight`); `s.Re_x`/`s.Re_y` stay `|q|/nu`.
"""
compute_face_flux!(s::State, p::ModelParameters, kfs::AbstractKFaceScheme = Arithmetic(), gc::AbstractGapCoupling = ExplicitGapCoupling()) =
    (compute_b_w!(s, p); compute_free_surface_weight!(s, p); @parallel compute_face_flux_kernel!(s.K_x, s.K_y, s.q_x, s.q_y, s.Re_x, s.Re_y, s.b_w, s.mask, s.dhdx, s.dhdy, s.valid_x, s.valid_y, p.g / (12 * p.nu), p.nu, p.omega, kfs, s.dhdx, s.dhdy, s.fsw, p.fs_factor - 1, p.fs_omega - 1); s)

# Under ImplicitUpwind the face flux q = -K(W_d)*dphi of a face depends on the head through the water depth W_d of
# its donor d (the higher-potential cell), where that depth is free (s.dbw_dh = 1: unfilled or uplifted). With
# K0 = c0*W^3 and Re from omega*Re^2 + Re - K0*|dphi|/nu = 0 (compute_face_flux_kernel!), the derivative of
# |q| = nu*Re at fixed dphi is d|q|/dW = 3*K0*|dphi|/(W*(1 + 2*omega*Re)) (3 K/W laminar, 3/2 K/W turbulent), and
# per head of the donor J = d|q|/dW * dW/dh [m/s] (the head row divides it by the face spacing). Jw_x[i] stores +J if the
# donor is the cell at the high index i and -J if it is the cell at i - 1; 0 if the donor's depth is not free, if it
# is not GROUNDED (inflow from an OCEAN/LAND cell: the face takes the grounded cell's conductance, see
# face_conductance, which this head row does not linearize), if the face carries no flux, or if its donor is frozen on
# the downstream side (update_donor_kernel!, or a DONOR_THINNER face whose thinner cell is downstream: there dF/dh_d
# has the opposite sign, which would break the M-matrix
# pattern; J ~ |dphi| is small at such a face anyway, and the face is iterated as a plain fixed point in W).
# With a free-surface weight w(h) of the donor (free_surface_weight), K0 = c0 (1 + a w) W^3 and omega_e = omega (1 + o w)
# (a = fs_factor - 1, o = fs_omega - 1) move with the head too: per head of the donor
#   d|q|/dh = [K (1 + omega_e Re) |dphi| (3 dW/dh / W + a dw/dh / (1 + a w)) - nu Re^2 omega o dw/dh] / (1 + 2 omega_e Re),
# which is 3 K/W dW/dh (1 + omega Re)/(1 + 2 omega Re) for w = 0. dw/dh <= 0 (filling turns the layer into a duct, 4 times
# less conductive), and where that outweighs the deeper water the flux falls as the head rises; J is then kept at 0
# (a negative J breaks the M-matrix pattern, as for a donor frozen downstream).
@inline function free_water_newton(K, Re, dphi, W, dW, omega, w, dw, a, o, nu)
    om = omega * (1 + o * w)
    J = (K * (1 + om * Re) * dphi * (3 * dW / W + a * dw / (1 + a * w)) - nu * Re^2 * omega * o * dw) / (1 + 2 * om * Re)
    return max(J, zero(J))
end
@parallel_indices (ix, iy) function water_depth_newton_kernel!(Jw_x, Jw_y, K_x, K_y, Re_x, Re_y, dhdx, dhdy, b_w, dbw_dh, mask, omega, don_x, don_y, fsw, dfsw_dh, fs_c, fs_o, nu)
    nx, ny = size(b_w, 1), size(b_w, 2)
    if ix <= size(Jw_x, 1) && iy <= size(Jw_x, 2)
        J = zero(omega)
        if ix > 1 && ix <= nx && K_x[ix, iy] > 0 && dhdx[ix, iy] != 0
            hi = donor_side(don_x[ix, iy], b_w[ix-1, iy], b_w[ix, iy]) > 0 # the donor is cell ix (update_donor_kernel!: the higher-potential cell, unless frozen)
            d = hi ? ix : ix - 1
            if (dhdx[ix, iy] > 0) == hi && mask[d, iy] == GROUNDED && dbw_dh[d, iy] > 0
                Re = Re_x[ix, iy]
                J = free_water_newton(K_x[ix, iy], Re, abs(dhdx[ix, iy]), b_w[d, iy], dbw_dh[d, iy], omega, fsw[d, iy], dfsw_dh[d, iy], fs_c, fs_o, nu)
                J = hi ? J : -J
            end
        end
        Jw_x[ix, iy] = J
    end
    if ix <= size(Jw_y, 1) && iy <= size(Jw_y, 2)
        J = zero(omega)
        if iy > 1 && iy <= ny && K_y[ix, iy] > 0 && dhdy[ix, iy] != 0
            hi = donor_side(don_y[ix, iy], b_w[ix, iy-1], b_w[ix, iy]) > 0
            d = hi ? iy : iy - 1
            if (dhdy[ix, iy] > 0) == hi && mask[ix, d] == GROUNDED && dbw_dh[ix, d] > 0
                Re = Re_y[ix, iy]
                J = free_water_newton(K_y[ix, iy], Re, abs(dhdy[ix, iy]), b_w[ix, d], dbw_dh[ix, d], omega, fsw[ix, d], dfsw_dh[ix, d], fs_c, fs_o, nu)
                J = hi ? J : -J
            end
        end
        Jw_y[ix, iy] = J
    end
    return
end

"""
$(TYPEDSIGNATURES)

[`ImplicitUpwind`](@ref) method: the face fluxes from the donor's water depth at the end of the step
([`compute_b_w_iterate!`](@ref), from the current head iterate and the gap coupling `gc`), and the Newton coefficients `s.Jw_x`/`s.Jw_y` of
their dependence on the donor's head through that depth (`water_depth_newton_kernel!`), which the head
rows add (`add_water_depth_newton!`, `linear_solver.jl`). The donor of a face is the higher-potential cell, except
that a face whose donor has flipped `MAX_DONOR_FLIPS` times within the step keeps its donor (`reset_donors!`).
"""
function compute_face_flux!(s::State, p::ModelParameters, kfs::ImplicitUpwind, gc::AbstractGapCoupling = ExplicitGapCoupling())
    compute_b_w_iterate!(s, p, gc)
    @parallel update_donor_kernel!(s.don_x, s.don_y, s.dhdx, s.dhdy, MAX_DONOR_FLIPS)
    compute_free_surface_weight!(s, p)
    @parallel compute_face_flux_kernel!(s.K_x, s.K_y, s.q_x, s.q_y, s.Re_x, s.Re_y, s.b_w, s.mask, s.dhdx, s.dhdy, s.valid_x, s.valid_y, p.g / (12 * p.nu), p.nu, p.omega, kfs, s.don_x, s.don_y, s.fsw, p.fs_factor - 1, p.fs_omega - 1)
    @parallel water_depth_newton_kernel!(s.Jw_x, s.Jw_y, s.K_x, s.K_y, s.Re_x, s.Re_y, s.dhdx, s.dhdy, s.b_w, s.dbw_dh, s.mask, p.omega, s.don_x, s.don_y, s.fsw, s.dfsw_dh, p.fs_factor - 1, p.fs_omega - 1, p.nu)
    return s
end

# Donor of each face under ImplicitUpwind, kept in s.don_x/s.don_y as sign * (1 + flips): the sign is the side of the
# donor (+1 the cell at the high index, -1 at the low index), flips counts how often it changed within this step (reset
# at the start of the step, reset_donors!). Normally the donor is the higher-potential cell, as in Upwind. A face whose
# donor has changed MAX_DONOR_FLIPS times keeps it for the rest of the step: where the flow reverses across a face
# between a thick layer and a thin one, the face conductance jumps by orders of magnitude with the sign of dphi, and
# Newton can cycle across that jump (seen on Drang Drung at dt = 1 h and 1 day: a face next to a blister flipping every
# iteration, K between 5e-4 and 30). The flux q = -K dphi itself is continuous at the reversal (it is 0 there); the
# frozen face just keeps the conductance of the earlier donor, the larger one in that case, so the error is at most that
# of a mean of the two conductances on that one face, and only where the flow direction is undecided. The water budget
# is unaffected (both cells see the same face flux).
# Once the head has converged, polish_donors! revisits the frozen faces: one whose flow ended up against its frozen
# donor (frozen on the downstream side) is refrozen on its upwind side, |don| = DONOR_REFROZEN, and the head iterated
# again; a refrozen face that reverses once more takes the conductance of the thinner of its two cells, |don| =
# DONOR_THINNER (the Upwind face mean at no flow, the smaller error of the two donors), and is left there. At most
# MAX_DONOR_POLISH_ROUNDS rounds per step. On Drang Drung (dt = 1 h to 1 day) this removed nearly all faces left frozen
# downstream at the end of a step (dt = 1 day: 31 of 364 steps with such faces -> 2, dt = 1 h: 11 of 8760 -> 0) at the
# same cost and with no failed steps; the year-end N moved by at most 0.01 MPa (0.04 MPa on the water-surface potential).
const MAX_DONOR_FLIPS = 4
const DONOR_REFROZEN = 1000 # |don| of a face refrozen on its upwind side by polish_donors!
const DONOR_THINNER = 2000  # |don| of a refrozen face that reversed again: the thinner cell's conductance
const MAX_DONOR_POLISH_ROUNDS = 3
@inline function next_donor(d, dh, maxflips)
    sd = dh > 0 ? one(d) : (dh < 0 ? -one(d) : zero(d))
    d == 0 && return sd                          # not set yet: the higher-potential cell (0 while there is no flow)
    flips = abs(d) - 1
    (flips >= maxflips || sd == 0 || sd == sign(d)) && return d
    return sd * (flips + 2)                      # the flow reversed: switch donor, count the flip
end
@parallel_indices (ix, iy) function update_donor_kernel!(don_x, don_y, dhdx, dhdy, maxflips)
    if ix <= size(don_x, 1) && iy <= size(don_x, 2)
        don_x[ix, iy] = next_donor(don_x[ix, iy], dhdx[ix, iy], maxflips)
    end
    if ix <= size(don_y, 1) && iy <= size(don_y, 2)
        don_y[ix, iy] = next_donor(don_y[ix, iy], dhdy[ix, iy], maxflips)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Clears the face donors of [`ImplicitUpwind`](@ref) (`s.don_x`, `s.don_y`) at the start of a step, so each step starts
from the higher-potential cell of every face and counts its donor flips afresh (see `update_donor_kernel!`). A no-op for
every other K-face scheme. Also zeroes the previous head steps of the oscillation damping (`s.dh_it`,
[`limit_head_step!`](@ref)), for every K-face scheme.
"""
reset_donors!(::AbstractKFaceScheme, s::State) = (fill!(s.dh_it, 0); s)
reset_donors!(::ImplicitUpwind, s::State) = (fill!(s.don_x, 0); fill!(s.don_y, 0); fill!(s.dh_it, 0); s)

# Side of the cell whose water depth sets the conductance of a face with donor d (+1 the cell at the high index, -1 at
# the low index, 0 while not set): the donor, or for a DONOR_THINNER face the thinner cell (Wa, Wb the low/high-index
# water depths; the low-index cell on a tie, as min in face_mean).
@inline donor_side(d, Wa, Wb) = abs(d) >= DONOR_THINNER ? (Wb < Wa ? one(d) : -one(d)) : sign(d)
# Donor of a frozen face after polishing at the converged head (dh the head gradient across it), see polish_donors!.
@inline function polished_donor(d, dh, maxflips)
    (d == 0 || dh == 0 || abs(d) - 1 < maxflips || abs(d) >= DONOR_THINNER || sign(d) == sign(dh)) && return d
    return sign(dh) * (abs(d) >= DONOR_REFROZEN ? DONOR_THINNER : DONOR_REFROZEN)
end
@parallel_indices (ix, iy) function polish_donor_kernel!(don_x, don_y, dhdx, dhdy, maxflips)
    if ix <= size(don_x, 1) && iy <= size(don_x, 2)
        don_x[ix, iy] = polished_donor(don_x[ix, iy], dhdx[ix, iy], maxflips)
    end
    if ix <= size(don_y, 1) && iy <= size(don_y, 2)
        don_y[ix, iy] = polished_donor(don_y[ix, iy], dhdy[ix, iy], maxflips)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Polishes the frozen face donors of [`ImplicitUpwind`](@ref) at a converged head and returns the number of faces changed
(0 for every other K-face scheme): a face frozen on its downstream side is refrozen on its upwind side, and one that
was refrozen and reversed again takes the conductance of its thinner cell (see the notes above `MAX_DONOR_FLIPS`).
The Picard loop ([`Picard_loop!`](@ref)) iterates the head again while this changes
faces, up to `MAX_DONOR_POLISH_ROUNDS` times per step.
"""
polish_donors!(::AbstractKFaceScheme, s::State) = 0
function polish_donors!(::ImplicitUpwind, s::State)
    changed(d, dh) = Int(polished_donor(d, dh, MAX_DONOR_FLIPS) != d)
    n = mapreduce(changed, +, s.don_x, s.dhdx; init = 0) + mapreduce(changed, +, s.don_y, s.dhdy; init = 0) # once per converged step, the two-array mapreduce's temporary is no concern here
    n > 0 && @parallel polish_donor_kernel!(s.don_x, s.don_y, s.dhdx, s.dhdy, MAX_DONOR_FLIPS)
    return n
end
