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
# fusion L. The four heat-source/sink terms are exposed as their own
# standalone kernels below (compute_shear!/compute_potential!/
# compute_sensible!, writing into preallocated State fields s.Q_b/
# s.Q_diss/s.Q_sens for standalone/diagnostic use, e.g. as a tracked_obs
# name -- see Simulation's tracked_obs; q_T has no standalone kernel since
# it's an input field, not something Shakti computes), but compute_mdot!'s
# hot path (below) does NOT call compute_shear!/compute_potential!/
# compute_sensible!: it uses its own fused kernel that recomputes the same
# terms and combines them into mdot in a single kernel launch instead of
# four, while still writing shear/potential/sensible so those fields stay
# valid every Picard iteration for anyone reading them (when the
# corresponding MeltTerms flag is on -- see compute_mdot_kernel! below).

@parallel_indices (ix, iy) function compute_shear_kernel!(Q_b, ub_x, taub_x, ub_y, taub_y)
    if ix <= size(Q_b, 1) && iy <= size(Q_b, 2)
        Q_b[ix, iy] = abs((ub_x[ix+1, iy]*taub_x[ix+1, iy] + ub_x[ix, iy]*taub_x[ix, iy]) / 2 +
                            (ub_y[ix, iy+1]*taub_y[ix, iy+1] + ub_y[ix, iy]*taub_y[ix, iy]) / 2)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.Q_b`: the frictional (sliding) heat `|u_b . taub|` [W/m^2], averaged from faces onto
cell centers. Standalone/diagnostic use (e.g. as a `tracked_obs` name,
see `Simulation`'s `tracked_obs`) -- [`compute_mdot!`](@ref)'s hot path recomputes this term
itself in a fused kernel rather than calling this function.
"""
compute_shear!(s::State) = (@parallel compute_shear_kernel!(s.Q_b, s.ub_x, s.taub_x, s.ub_y, s.taub_y); s)

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

@parallel_indices (ix, iy) function compute_potential_kernel!(Q_diss, mask, q_x, dhdx, q_y, dhdy, rho_w, ggrav)
    if ix <= size(Q_diss, 1) && iy <= size(Q_diss, 2)
        Q_diss[ix, iy] = rho_w * ggrav * abs(cell_face_sum(mask, q_x, dhdx, q_y, dhdy, ix, iy))
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.Q_diss`: the heat dissipated by water flowing down the hydraulic-head gradient,
`rho_w*g*|q . grad(h)|` [W/m^2], averaged from faces onto cell centers.
Standalone/diagnostic use, same caveat as [`compute_shear!`](@ref).
"""
compute_potential!(s::State, p::ModelParameters) = (@parallel compute_potential_kernel!(s.Q_diss, s.mask, s.q_x, s.dhdx, s.q_y, s.dhdy, p.rho_w, p.g); s)

# Requires dpwdx/dpwdy (computed above) already current.
@parallel_indices (ix, iy) function compute_sensible_kernel!(Q_sens, mask, q_x, dpwdx, q_y, dpwdy, ct, cw, rho_w)
    if ix <= size(Q_sens, 1) && iy <= size(Q_sens, 2)
        Q_sens[ix, iy] = ct * cw * rho_w * cell_face_sum(mask, q_x, dpwdx, q_y, dpwdy, ix, iy)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.Q_sens`: the sensible heat exchanged as water moves to regions of different
pressure-melting-point temperature, `ct*cw*rho_w*(q . grad(pw))` [W/m^2], averaged from faces onto
cell centers. Requires `s.dpwdx`/`s.dpwdy` already current. Standalone/diagnostic use, same
caveat as [`compute_shear!`](@ref).
"""
compute_sensible!(s::State, p::ModelParameters) = (@parallel compute_sensible_kernel!(s.Q_sens, s.mask, s.q_x, s.dpwdx, s.q_y, s.dpwdy, p.ct, p.cw, p.rho_w); s)

# Fused hot-path kernel: Q_b/Q_diss/Q_sens/mdot in one launch instead of
# four. Duplicates the per-cell math above rather than calling those
# kernels, since each is itself a separate kernel launch. Each term is
# stored as a heat flux [W/m^2] with its prefactor applied, so the stored
# fields are exactly what enters mdot and can be passed to a coupled ice
# model as is. A term whose MeltTerms flag is off is not computed (its
# inputs are never read) and its field is set to 0, so it never holds a
# stale value.
@parallel_indices (ix, iy) function compute_mdot_kernel!(mdot, mask, Q_b, Q_diss, Q_sens, G, q_T, ub_x, taub_x, ub_y, taub_y, q_x, dhdx, q_y, dhdy, dpwdx, dpwdy, Linv, rho_w, ggrav, ct, cw,
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

        mdot[ix, iy] = Linv * acc
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
stored as 0. Computed in one fused kernel rather than by calling [`compute_shear!`](@ref) etc. (one
launch instead of four). The englacial input `s.ieb` is not part of `mdot`: it enters the water
mass balance only, not the gap opening.

Dispatches on `sim.mt` (decided once in `Simulation`'s constructor from `p.mdot_includes_G`/
`p.mdot_includes_frictional`/`p.mdot_includes_potential`/`p.mdot_includes_sensible`/
`p.mdot_includes_qT`): a term whose flag is `false` never touches its own inputs/output field at
all (e.g. `s.dpwdx`/`s.dpwdy` when `Sensible == false`), rather than computing it and multiplying by
a zero prefactor; its output field is set to 0.
"""
function compute_mdot!(s::State, p::ModelParameters, mt::MeltTerms)
    @parallel compute_mdot_kernel!(s.mdot, s.mask, s.Q_b, s.Q_diss, s.Q_sens, s.G, s.q_T, s.ub_x, s.taub_x, s.ub_y, s.taub_y, s.q_x, s.dhdx, s.q_y, s.dhdy, s.dpwdx, s.dpwdy, 1/p.L, p.rho_w, p.g, p.ct, p.cw, mt)
    return s
end

# D (SUHMO Eq. 10) is built from exactly the same two dissipation terms as mdot's own
# Potential/Sensible contributions (potential-energy dissipation, sensible-heat exchange) -- just
# with a different prefactor (b/(rho_i*L) instead of 1/L) and no geothermal/frictional/conductive
# terms (those have no role in melting a channel's side walls). Reusing mt's own type parameters
# to gate them, rather than a fresh always-on formula, keeps D consistent with whatever a run
# already chose for mdot: a run with Sensible off shouldn't have D silently include it just
# because p.ct/p.cw happen to be nonzero -- same rationale as ModelParameters' own docstring note
# on mdot_includes_sensible not being inferred from ct/cw.
#
# D is floored at 0. D models channel walls melted by *surplus* dissipated heat; with the
# sensible-heat (pressure-melting) term on, water flowing up a steep adverse bed slope can have a
# net heat deficit (supercooling), which would make D negative -- a backward-diffusion operator,
# ill-posed and no longer SPD. The deficit is already represented locally by mdot going negative
# (freeze-on), so walls just stop melting there. A no-op whenever ct == 0 (SUHMO's own choice in
# nearly all of Felden et al. 2023's experiments).
#
# Both faces in one launch (ParallelStencil infers the launch range from the union of every
# argument's size, see compute_dhdxy_kernel!'s own note, field_gradients.jl); D_x/D_y are left
# exactly zero (their @zeros default, never written) at domain-boundary faces -- via this kernel's
# own explicit `ix > 1 && ix < size(D_x, 1)` / `iy > 1 && iy < size(D_y, 2)` guard, same convention
# as compute_dhdx_kernel!/compute_dhdy_kernel! (field_gradients.jl) -- and at any face touching
# OTHER_BASIN/FROZEN_BED, but there NOT via an explicit mask branch: those faces are built from
# q_x/dhdx/dpwdx (and the y-face equivalents), which are themselves already exactly zero there
# (see compute_dhdx_kernel!'s docstring): D_x = f(q_x, dhdx, dpwdx), and q_x itself comes out to 0
# wherever dhdx=0 (compute_q_and_Re_x_kernel!, water_flux.jl), so the zero propagates through
# automatically.
@parallel_indices (ix, iy) function compute_D_kernel!(D_x, D_y, b_x, b_y, q_x, q_y, dhdx, dhdy, dpwdx, dpwdy, Linv, rho_w, rho_i, ggrav, ct, cw,
                                                        ::MeltTerms{Geothermal,Frictional,Potential,Sensible,Conductive}) where {Geothermal,Frictional,Potential,Sensible,Conductive}
    if ix > 1 && ix < size(D_x, 1) && iy <= size(D_x, 2)
        acc = zero(eltype(D_x))
        if Potential
            acc -= rho_w * ggrav * q_x[ix, iy] * dhdx[ix, iy]
        end
        if Sensible
            acc += ct * cw * rho_w * q_x[ix, iy] * dpwdx[ix, iy]
        end
        D_x[ix, iy] = max(zero(acc), (b_x[ix, iy] / rho_i) * Linv * acc) # floored at 0, see below
    end
    if iy > 1 && iy < size(D_y, 2) && ix <= size(D_y, 1)
        acc = zero(eltype(D_y))
        if Potential
            acc -= rho_w * ggrav * q_y[ix, iy] * dhdy[ix, iy]
        end
        if Sensible
            acc += ct * cw * rho_w * q_y[ix, iy] * dpwdy[ix, iy]
        end
        D_y[ix, iy] = max(zero(acc), (b_y[ix, iy] / rho_i) * Linv * acc)
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

Updates `s.D_x`/`s.D_y` (the channel-wall diffusion coefficient, SUHMO Eq. 10) from the current
`s.b_x`/`s.b_y`/`s.q_x`/`s.q_y`/`s.dhdx`/`s.dhdy`/`s.dpwdx`/`s.dpwdy` under [`WithDiffusion`](@ref)
(`linear_solver.jl`) -- its two terms (potential-energy dissipation, sensible-heat exchange) gated
by `mt`'s own `Potential`/`Sensible` flags, exactly like [`compute_mdot!`](@ref)'s matching terms.
"""
function compute_D!(s::State, p::ModelParameters, mt::MeltTerms, ::WithDiffusion)
    @parallel compute_D_kernel!(s.D_x, s.D_y, s.b_x, s.b_y, s.q_x, s.q_y, s.dhdx, s.dhdy, s.dpwdx, s.dpwdy, 1/p.L, p.rho_w, p.rho_i, p.g, p.ct, p.cw, mt)
    return s
end
