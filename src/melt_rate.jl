"""
$(TYPEDSIGNATURES)

Which terms [`compute_mdot!`](@ref) includes -- one `Bool` type parameter per term (`Geothermal`,
`Frictional`, `Potential`, `Sensible`, `Conductive`), decided ONCE in `Simulation`'s constructor
from `p.mdot_includes_G`/`p.mdot_includes_frictional`/`p.mdot_includes_potential`/
`p.mdot_includes_sensible`/`p.mdot_includes_qT` (see `simulation.jl`) rather than every call, so
turning a term off skips its kernel work and array traffic entirely instead of just multiplying its
contribution by zero: a `Bool` *type* parameter (not a runtime field read inside the kernel) is
resolved by the compiler when `compute_mdot_kernel!` is specialized for a concrete `MeltTerms{...}`,
so `if Geothermal ... end` etc. below are dead-code-eliminated per term per specialization, not
branched on at runtime -- same "dispatch on a type decided once outside the hot loop" idiom as
`canonical_exponent`/`pow` (`model_parameters.jl`) and the head/gap schemes (`simulation.jl`), just
with N independent flags bundled into one type instead of one type per flag (which doesn't scale:
N independently-toggleable terms would otherwise need N type parameters threaded through
`Simulation`/`elliptic_solver!`/`Picard_loop!`/`Picard_iteration!`, or 2^N hand-written kernel
bodies to keep the "skip entirely when off" property).

# Notes

Defined here, next to their sole consumer ([`compute_mdot!`](@ref) below), rather than in
`simulation.jl`: `elliptic_solver.jl` (included before `simulation.jl` for its own
`PicardSolver`/`Simulation` ordering reasons) needs `MeltTerms` to type-annotate `mt`, so this file
is included before `elliptic_solver.jl` too.
"""
struct MeltTerms{Geothermal,Frictional,Potential,Sensible,Conductive} end

# Melt rate = geothermal flux + frictional (sliding) heating + potential
# energy released by water flowing downgradient + sensible heat exchanged as
# water moves to regions of different pressure melting point - conductive
# heat lost into cold ice above the bed, all divided by the latent heat of
# fusion L. compute_mdot! computes the three heat terms (frictional,
# dissipation, sensible; G and q_T are input fields) and mdot in one fused
# kernel, writing each term to its State field (s.Q_b/s.Q_diss/s.Q_sens) so
# those fields hold the current values every Picard iteration (when the
# corresponding MeltTerms flag is on -- see compute_mdot_kernel! below), e.g.
# for a tracked_obs name.

# The frictional heat takes each component as the mean of its two faces, also at a face to an OCEAN/LAND cell, unlike the
# dissipation terms (face_heat_share below): it is a heat density of the grounded bed, not the energy released along the
# path between two cell centres. Its stress at that face is already the grounded cell's (face_x/face_y, sliding_law.jl),
# so the mean holds the grounded value up to the face, and beyond it (floating ice, land) there is no friction to credit.
# With uniform sliding the mean gives u_b.taub exactly; the full share of face_heat_share would give 1.5 times it.
# Each computed law gives a stress along u_b with N floored at 0 (bed_N, sliding_law.jl), so u_b.taub >= 0 and abs() is
# a no-op for them; it is kept for PrescribedSlidingLaw, whose taub comes from the host with its own sign convention.

# Share of a face's dissipation (q.grad(h) or q.grad(pw) on that face) credited to cell (ix, iy):
# half, as each interior face is split evenly between its two cells -- except the face between a
# GROUNDED cell and an OCEAN/LAND (Dirichlet) cell, whose Dirichlet side has no hydrology to
# receive its half, so the grounded cell takes all of it (otherwise half the heat dissipated by
# the outlet flow would silently vanish). (jx, jy) is the neighbour across the face; faces on the
# domain edge carry zero flux, so their share is irrelevant.
@inline function face_heat_share(mask, ix, iy, jx, jy)
    inside = (1 <= jx <= size(mask, 1)) & (1 <= jy <= size(mask, 2))
    full = inside && (mask[ix, iy] == GROUNDED) && is_dirichlet(mask[jx, jy])
    return full ? one(eltype(mask)) : one(eltype(mask)) / 2
end

# Face-to-cell sum of a face product f_x*g_x + f_y*g_y, each face weighted by face_heat_share.
@inline function cell_face_sum(mask, fx, gx, fy, gy, ix, iy)
    return face_heat_share(mask, ix, iy, ix+1, iy) * fx[ix+1, iy] * gx[ix+1, iy] +
           face_heat_share(mask, ix, iy, ix-1, iy) * fx[ix, iy]   * gx[ix, iy] +
           face_heat_share(mask, ix, iy, ix, iy+1) * fy[ix, iy+1] * gy[ix, iy+1] +
           face_heat_share(mask, ix, iy, ix, iy-1) * fy[ix, iy]   * gy[ix, iy]
end

# Fused hot-path kernel: Q_b/Q_diss/Q_sens/mdot in one launch instead of
# four. Each term is stored as a heat flux [W/m^2] with its prefactor
# applied, so the stored fields are exactly what enters mdot and can be passed to a coupled ice
# model as is. A term whose MeltTerms flag is off is not computed (its
# inputs are never read) and its field is set to 0, so it never holds a
# stale value.
@parallel_indices (ix, iy) function compute_mdot_kernel!(mdot, mdot_min, limit_freeze_on, mask, Q_b, Q_diss, Q_sens, G, q_T, ub_x, taub_x, ub_y, taub_y, q_x, dhdx, q_y, dhdy, dpwdx, dpwdy, Linv, rho_w, ggrav, ct, cw,
                                                          ::MeltTerms{Geothermal,Frictional,Potential,Sensible,Conductive}) where {Geothermal,Frictional,Potential,Sensible,Conductive}
    if ix <= size(mdot, 1) && iy <= size(mdot, 2)
        acc = zero(eltype(mdot))

        if Geothermal
            acc += G[ix, iy]
        end

        if Frictional
            sh = abs((ub_x[ix+1, iy]*taub_x[ix+1, iy] + ub_x[ix, iy]*taub_x[ix, iy]) / 2 +
                     (ub_y[ix, iy+1]*taub_y[ix, iy+1] + ub_y[ix, iy]*taub_y[ix, iy]) / 2)
            Q_b[ix, iy] = sh
            acc += sh
        else
            Q_b[ix, iy] = zero(acc)
        end

        if Potential
            pot = rho_w*ggrav*abs(cell_face_sum(mask, q_x, dhdx, q_y, dhdy, ix, iy))
            Q_diss[ix, iy] = pot
            acc += pot
        else
            Q_diss[ix, iy] = zero(acc)
        end

        if Sensible
            sens = ct*cw*rho_w*cell_face_sum(mask, q_x, dpwdx, q_y, dpwdy, ix, iy)
            Q_sens[ix, iy] = sens
            acc += sens
        else
            Q_sens[ix, iy] = zero(acc)
        end

        if Conductive
            acc -= q_T[ix, iy]
        end

        # Freeze-on is limited to the water the gap can supply this step: once the gap is at its
        # floor there is nothing left to freeze, and the remaining heat deficit is the host's (it
        # cools the ice), not water expelled into the flow.
        mdot[ix, iy] = limit_freeze_on ? max(Linv * acc, mdot_min[ix, iy]) : Linv * acc
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.mdot` (subglacial melt rate) = geothermal flux `s.G` + frictional (sliding) heating +
potential energy released by water flowing downgradient + sensible heat exchanged as water moves to
regions of different pressure melting point - conductive heat lost into cold ice above the bed
`s.q_T`, all divided by the latent heat of fusion `p.L` -- each term included or not per `mt`'s type
parameters (see [`MeltTerms`](@ref)). Also stores the heat terms `s.Q_b`/`s.Q_diss`/`s.Q_sens`
[W/m^2], prefactors applied, so `L*mdot = G + Q_b + Q_diss + Q_sens - q_T`; a term that is off is
stored as 0. With `p.limit_freeze_on`, `mdot` is floored at `s.mdot_min = -rho_i*C`, the
freeze-on the gap can supply this step ([`freeze_on_capacity!`](@ref), set once per step by
`prepare_head_solve!`); the heat terms are stored unlimited. Computed in one fused kernel (one
launch instead of four). The englacial input `s.ieb` is not part of `mdot`: it enters the water
mass balance only, not the gap opening.

Dispatches on `sim.mt` (decided once in `Simulation`'s constructor from `p.mdot_includes_G`/
`p.mdot_includes_frictional`/`p.mdot_includes_potential`/`p.mdot_includes_sensible`/
`p.mdot_includes_qT`): a term whose flag is `false` never touches its own inputs/output field at
all (e.g. `s.dpwdx`/`s.dpwdy` when `Sensible == false`), rather than computing it and multiplying by
a zero prefactor; its output field is set to 0.
"""
function compute_mdot!(s::State, p::ModelParameters, mt::MeltTerms)
    @parallel compute_mdot_kernel!(s.mdot, s.mdot_min, p.limit_freeze_on, s.mask, s.Q_b, s.Q_diss, s.Q_sens, s.G, s.q_T, s.ub_x, s.taub_x, s.ub_y, s.taub_y, s.q_x, s.dhdx, s.q_y, s.dhdy, s.dpwdx, s.dpwdy, 1/p.L, p.rho_w, p.g, p.ct, p.cw, mt)
    return s
end

# D (SUHMO Eq. 10) is one scalar per cell, D = min(W, b)/(rho_i L) * (Q_diss + Q_sens), built from exactly
# the same two dissipation terms as mdot's own Potential/Sensible contributions (potential-energy dissipation,
# sensible-heat exchange; cell_face_sum, as compute_mdot_kernel! sums them) -- just with a different prefactor
# (b/(rho_i*L) instead of 1/L) and no geothermal/frictional/conductive terms (those have no role in melting a
# channel's side walls). It is then averaged onto the faces. Building it per face from that face's own flux
# (D_y from q_y only) gave a channel along x no sideways diffusion at all (D_y 6-9 orders too small) and made
# the result depend on the grid orientation. The b of the prefactor is the height of wall the water touches:
# the filled part of the gap, min(W, b) (wet_gap below, W = State.W the water), which is b except in an unfilled cell
# (UnfilledCavities), where only the bottom W of the gap holds water. Reusing mt's own type
# parameters to gate the terms, rather than a fresh always-on formula, keeps D consistent with whatever a run
# already chose for mdot: a run with Sensible off shouldn't have D silently include it just because p.ct/p.cw
# happen to be nonzero -- same rationale as ModelParameters' own docstring note on mdot_includes_sensible not
# being inferred from ct/cw.
#
# D is floored at 0 per cell. D models channel walls melted by *surplus* dissipated heat; with the
# sensible-heat (pressure-melting) term on, water flowing up a steep adverse bed slope can have a
# net heat deficit (supercooling), which would make D negative -- a backward-diffusion operator,
# ill-posed and no longer SPD. The deficit is already represented locally by mdot going negative
# (freeze-on), so walls just stop melting there. A no-op whenever ct == 0 (SUHMO's own choice in
# nearly all of Felden et al. 2023's experiments).
#
# Both faces in one launch (ParallelStencil infers the launch range from the union of every argument's size, see
# compute_dhdxy_kernel!'s own note, field_gradients.jl). D_x/D_y are exactly zero at domain-boundary faces (never
# written, their @zeros default) and at every face that does not join two GROUNDED cells: only those faces carry
# the diffusion (diffusion_source, linear_solver.jl). Each cell's D is recomputed by both faces that read it,
# which is cheaper than a cell-sized scratch field for a stencil this small.
# Filled part of the gap, the water W kept within [0, b]: a filled cell may hold more water than its gap (the uplift
# layer, or the water the budget keeps above the gap, update_water_budget!), and the wall the water touches is at most b.
@inline wet_gap(b, W) = clamp(W, zero(b), b)

@inline function cell_heat(mask, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, rho_w, ggrav, ct, cw, ix, iy,
                           ::MeltTerms{Geothermal,Frictional,Potential,Sensible,Conductive}) where {Geothermal,Frictional,Potential,Sensible,Conductive}
    heat = zero(rho_w)
    if Potential
        heat += rho_w * ggrav * abs(cell_face_sum(mask, q_x, dhdx, q_y, dhdy, ix, iy))
    end
    if Sensible
        heat += ct * cw * rho_w * cell_face_sum(mask, q_x, dpwdx, q_y, dpwdy, ix, iy)
    end
    return heat
end

@inline cell_D(mask, b, W, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, scale, rho_w, ggrav, ct, cw, ix, iy, mt) =
    max(zero(scale), wet_gap(b[ix, iy], W[ix, iy]) * scale * cell_heat(mask, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, rho_w, ggrav, ct, cw, ix, iy, mt)) # floored at 0, see above

# D on the face between cells (ix, iy) and (jx, jy): the mean of the two cells' D, or with `D_faces` SUHMO's form, the mean
# gap of the two cells times the mean of their heats (each floored at 0), which is a quarter of a channel cell's D at the face
# to a dry neighbour where the mean of the cells' D is a half.
@inline function face_D(D_faces, mask, b, W, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, scale, rho_w, ggrav, ct, cw, ix, iy, jx, jy, D_min, mt)
    if D_faces
        hi = max(zero(scale), cell_heat(mask, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, rho_w, ggrav, ct, cw, ix, iy, mt))
        hj = max(zero(scale), cell_heat(mask, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, rho_w, ggrav, ct, cw, jx, jy, mt))
        bf = (wet_gap(b[ix, iy], W[ix, iy]) + wet_gap(b[jx, jy], W[jx, jy])) / 2
        return max(bf * scale * (hi + hj) / 2, D_min)
    end
    return max((cell_D(mask, b, W, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, scale, rho_w, ggrav, ct, cw, ix, iy, mt) +
                cell_D(mask, b, W, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, scale, rho_w, ggrav, ct, cw, jx, jy, mt)) / 2, D_min)
end

@parallel_indices (ix, iy) function compute_D_kernel!(D_x, D_y, mask, b, W, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, scale, rho_w, ggrav, ct, cw, D_min, D_faces, mt)
    if ix > 1 && ix < size(D_x, 1) && iy <= size(D_x, 2)
        D_x[ix, iy] = (mask[ix, iy] == GROUNDED && mask[ix-1, iy] == GROUNDED) ?
            face_D(D_faces, mask, b, W, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, scale, rho_w, ggrav, ct, cw, ix, iy, ix-1, iy, D_min, mt) : zero(scale)
    end
    if iy > 1 && iy < size(D_y, 2) && ix <= size(D_y, 1)
        D_y[ix, iy] = (mask[ix, iy] == GROUNDED && mask[ix, iy-1] == GROUNDED) ?
            face_D(D_faces, mask, b, W, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, scale, rho_w, ggrav, ct, cw, ix, iy, ix, iy-1, D_min, mt) : zero(scale)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Skips updating `s.D_x`/`s.D_y` entirely under [`NoDiffusion`](@ref) (`linear_solver.jl`) -- they
stay at whatever they last held (their `@zeros` default if never touched), which is correct since
nothing reads them when diffusion is off.
"""
compute_D!(s::State, p::ModelParameters, mt::MeltTerms, ::NoDiffusion) = s

"""
$(TYPEDSIGNATURES)

Updates `s.D_x`/`s.D_y` (the channel-wall diffusion coefficient, SUHMO Eq. 10) under [`WithDiffusion`](@ref)
(`linear_solver.jl`): one scalar per cell, `D = min(W, b)/(rho_i L) * (Q_diss + Q_sens)` floored at 0 (and at `ds.D_min` on a face between two `GROUNDED` cells, default 0), from the
current `s.b`/`s.W` (the filled part of the gap) and `s.q_x`/`s.q_y`/`s.dhdx`/`s.dhdy`/`s.dpwdx`/`s.dpwdy`, averaged
onto the faces between two `GROUNDED` cells (zero elsewhere), so it is the same across and along a channel -- its two terms (potential-energy dissipation, sensible-heat exchange) gated
by `mt`'s own `Potential`/`Sensible` flags, exactly like [`compute_mdot!`](@ref)'s matching terms.
"""
function compute_D!(s::State, p::ModelParameters, mt::MeltTerms, ds::WithDiffusion)
    @parallel compute_D_kernel!(s.D_x, s.D_y, s.mask, s.b, s.W, s.q_x, s.q_y, s.dhdx, s.dhdy, s.dpwdx, s.dpwdy, 1 / (p.rho_i * p.L), p.rho_w, p.g, p.ct, p.cw, convert(eltype(s.D_x), ds.D_min), ds.D_faces, mt)
    return s
end
