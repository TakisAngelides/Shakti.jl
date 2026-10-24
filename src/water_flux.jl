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

"""
$(TYPEDSIGNATURES)

Deprecated (removed in 0.3): [`compute_face_flux!`](@ref) with the default [`Arithmetic`](@ref) faces. Kept for old drivers that
rebuilt the initial fields themselves; [`set_initial_conditions!`](@ref) already does it.
"""
function compute_q_and_Re_xy!(s::State, p::ModelParameters)
    @warn "compute_q_and_Re_xy! is deprecated and will be removed in 0.3: use compute_face_flux!(s, p) (set_initial_conditions! already calls it)" maxlog = 1
    return compute_face_flux!(s, p)
end

"""
$(TYPEDSIGNATURES)

Deprecated (removed in 0.3): updates the cell-centred `s.K` (a diagnostic) only. The face transmissivities `s.K_x`/`s.K_y` the
linear system assembles are written by [`compute_face_flux!`](@ref); `kfs` is ignored.
"""
function compute_K!(s::State, p::ModelParameters, kfs::AbstractKFaceScheme = Arithmetic())
    @warn "compute_K! is deprecated and will be removed in 0.3: the face transmissivities come from compute_face_flux!, the cell-centred K from set_initial_conditions!" maxlog = 1
    return compute_cell_K!(s, p)
end

# Cell-centred K = b^3*g / (12*nu*(1 + omega*Re)), a diagnostic: the linear system assembles the face values
# K_x/K_y written by compute_face_flux!.
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
@parallel_indices (ix, iy) function compute_face_flux_kernel!(K_x, K_y, q_x, q_y, Re_x, Re_y, b, mask, dhdx, dhdy, valid_x, valid_y, c0, nu, omega, kfs::AbstractKFaceScheme, dsel_x, dsel_y)
    nx, ny = size(b, 1), size(b, 2)
    if ix <= size(K_x, 1) && iy <= size(K_x, 2)
        K0 = (ix > 1 && ix <= nx) ? face_conductance(kfs, c0 * b[ix-1, iy]^3, mask[ix-1, iy], c0 * b[ix, iy]^3, mask[ix, iy], face_select(kfs, dsel_x[ix, iy])) * valid_x[ix, iy] : zero(c0)
        D = K0 * abs(dhdx[ix, iy]) / nu
        Re = 2 * D / (1 + sqrt(1 + 4 * omega * D))
        Kf = K0 / (1 + omega * Re)
        K_x[ix, iy] = Kf
        Re_x[ix, iy] = Re
        q_x[ix, iy] = -Kf * dhdx[ix, iy]
    end
    if ix <= size(K_y, 1) && iy <= size(K_y, 2)
        K0 = (iy > 1 && iy <= ny) ? face_conductance(kfs, c0 * b[ix, iy-1]^3, mask[ix, iy-1], c0 * b[ix, iy]^3, mask[ix, iy], face_select(kfs, dsel_y[ix, iy])) * valid_y[ix, iy] : zero(c0)
        D = K0 * abs(dhdy[ix, iy]) / nu
        Re = 2 * D / (1 + sqrt(1 + 4 * omega * D))
        Kf = K0 / (1 + omega * Re)
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
`Re = 2*D / (1 + sqrt(1 + 4*omega*D))` (see the notes at the top of `water_flux.jl`).
"""
compute_face_flux!(s::State, p::ModelParameters, kfs::AbstractKFaceScheme = Arithmetic(), gc::AbstractGapCoupling = ExplicitGapCoupling()) =
    (compute_b_w!(s, p); @parallel compute_face_flux_kernel!(s.K_x, s.K_y, s.q_x, s.q_y, s.Re_x, s.Re_y, s.b_w, s.mask, s.dhdx, s.dhdy, s.valid_x, s.valid_y, p.g / (12 * p.nu), p.nu, p.omega, kfs, s.dhdx, s.dhdy); s)

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
@parallel_indices (ix, iy) function water_depth_newton_kernel!(Jw_x, Jw_y, K_x, K_y, Re_x, Re_y, dhdx, dhdy, b_w, dbw_dh, mask, omega, don_x, don_y)
    nx, ny = size(b_w, 1), size(b_w, 2)
    if ix <= size(Jw_x, 1) && iy <= size(Jw_x, 2)
        J = zero(omega)
        if ix > 1 && ix <= nx && K_x[ix, iy] > 0 && dhdx[ix, iy] != 0
            hi = donor_side(don_x[ix, iy], b_w[ix-1, iy], b_w[ix, iy]) > 0 # the donor is cell ix (update_donor_kernel!: the higher-potential cell, unless frozen)
            d = hi ? ix : ix - 1
            if (dhdx[ix, iy] > 0) == hi && mask[d, iy] == GROUNDED && dbw_dh[d, iy] > 0
                Re = Re_x[ix, iy]
                J = 3 * K_x[ix, iy] * (1 + omega * Re) / (b_w[d, iy] * (1 + 2 * omega * Re)) * abs(dhdx[ix, iy]) * dbw_dh[d, iy]
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
                J = 3 * K_y[ix, iy] * (1 + omega * Re) / (b_w[ix, d] * (1 + 2 * omega * Re)) * abs(dhdy[ix, iy]) * dbw_dh[ix, d]
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
    @parallel compute_face_flux_kernel!(s.K_x, s.K_y, s.q_x, s.q_y, s.Re_x, s.Re_y, s.b_w, s.mask, s.dhdx, s.dhdy, s.valid_x, s.valid_y, p.g / (12 * p.nu), p.nu, p.omega, kfs, s.don_x, s.don_y)
    @parallel water_depth_newton_kernel!(s.Jw_x, s.Jw_y, s.K_x, s.K_y, s.Re_x, s.Re_y, s.dhdx, s.dhdy, s.b_w, s.dbw_dh, s.mask, p.omega, s.don_x, s.don_y)
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
