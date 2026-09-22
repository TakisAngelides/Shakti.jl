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
# compute_sensible!, writing into preallocated State fields s.shear/
# s.potential/s.sensible for standalone/diagnostic use, e.g. as a tracked_obs
# name -- see Simulation's tracked_obs; q_T has no standalone kernel since
# it's an input field, not something Shakti computes), but compute_mdot!'s
# hot path (below) does NOT call compute_shear!/compute_potential!/
# compute_sensible!: it uses its own fused kernel that recomputes the same
# terms and combines them into mdot in a single kernel launch instead of
# four, while still writing shear/potential/sensible so those fields stay
# valid every Picard iteration for anyone reading them (when the
# corresponding MeltTerms flag is on -- see compute_mdot_kernel! below).

@parallel_indices (ix, iy) function compute_shear_kernel!(shear, ub_x, taub_x, ub_y, taub_y)
    if ix <= size(shear, 1) && iy <= size(shear, 2)
        shear[ix, iy] = abs((ub_x[ix+1, iy]*taub_x[ix+1, iy] + ub_x[ix, iy]*taub_x[ix, iy]) / 2 +
                            (ub_y[ix, iy+1]*taub_y[ix, iy+1] + ub_y[ix, iy]*taub_y[ix, iy]) / 2)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.shear`: the frictional (sliding) heating contribution to the melt rate, `|u_b . taub|`
averaged from faces onto cell centers. Standalone/diagnostic use (e.g. as a `tracked_obs` name,
see `Simulation`'s `tracked_obs`) -- [`compute_mdot!`](@ref)'s hot path recomputes this term
itself in a fused kernel rather than calling this function.
"""
compute_shear!(s::State) = (@parallel compute_shear_kernel!(s.shear, s.ub_x, s.taub_x, s.ub_y, s.taub_y); s)

@parallel_indices (ix, iy) function compute_potential_kernel!(potential, q_x, dhdx, q_y, dhdy)
    if ix <= size(potential, 1) && iy <= size(potential, 2)
        potential[ix, iy] = abs((q_x[ix+1, iy]*dhdx[ix+1, iy] + q_x[ix, iy]*dhdx[ix, iy]) / 2 +
                                (q_y[ix, iy+1]*dhdy[ix, iy+1] + q_y[ix, iy]*dhdy[ix, iy]) / 2)
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.potential`: the potential-energy-dissipation contribution to the melt rate (water
flowing down the hydraulic-head gradient), `|q . dhdx|` averaged from faces onto cell centers.
Standalone/diagnostic use, same caveat as [`compute_shear!`](@ref).
"""
compute_potential!(s::State) = (@parallel compute_potential_kernel!(s.potential, s.q_x, s.dhdx, s.q_y, s.dhdy); s)

# Requires dpwdx/dpwdy (computed above) already current.
@parallel_indices (ix, iy) function compute_sensible_kernel!(sensible, q_x, dpwdx, q_y, dpwdy)
    if ix <= size(sensible, 1) && iy <= size(sensible, 2)
        sensible[ix, iy] = (q_x[ix+1, iy]*dpwdx[ix+1, iy] + q_x[ix, iy]*dpwdx[ix, iy]) / 2 +
                            (q_y[ix, iy+1]*dpwdy[ix, iy+1] + q_y[ix, iy]*dpwdy[ix, iy]) / 2
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.sensible`: the sensible-heat-exchange contribution to the melt rate (water moving to
regions of different pressure-melting-point temperature), `q . dpwdx` averaged from faces onto
cell centers. Requires `s.dpwdx`/`s.dpwdy` already current. Standalone/diagnostic use, same
caveat as [`compute_shear!`](@ref).
"""
compute_sensible!(s::State) = (@parallel compute_sensible_kernel!(s.sensible, s.q_x, s.dpwdx, s.q_y, s.dpwdy); s)

# Fused hot-path kernel: shear/potential/sensible/mdot in one launch instead
# of four. Duplicates the per-cell math above rather than calling those
# kernels, since each is itself a separate kernel launch; still writes
# shear/potential/sensible (not just mdot) so those fields aren't left stale
# for anything that reads them after a Picard iteration -- UNLESS the
# corresponding MeltTerms flag is off, in which case that field is left
# untouched entirely (same "not just multiplied by zero" idiom the
# sensible-heat term already had, now applied uniformly to every term).
@parallel_indices (ix, iy) function compute_mdot_kernel!(mdot, shear, potential, sensible, G, q_T, ub_x, taub_x, ub_y, taub_y, q_x, dhdx, q_y, dhdy, dpwdx, dpwdy, Linv, rho_w, ggrav, ct, cw,
                                                          ::MeltTerms{Geothermal,Frictional,Potential,Sensible,Conductive}) where {Geothermal,Frictional,Potential,Sensible,Conductive}
    if ix <= size(mdot, 1) && iy <= size(mdot, 2)
        acc = zero(eltype(mdot))

        if Geothermal
            acc += G[ix, iy]
        end

        if Frictional
            sh = abs((ub_x[ix+1, iy]*taub_x[ix+1, iy] + ub_x[ix, iy]*taub_x[ix, iy]) / 2 +
                     (ub_y[ix, iy+1]*taub_y[ix, iy+1] + ub_y[ix, iy]*taub_y[ix, iy]) / 2)
            shear[ix, iy] = sh
            acc += sh
        end

        if Potential
            pot = abs((q_x[ix+1, iy]*dhdx[ix+1, iy] + q_x[ix, iy]*dhdx[ix, iy]) / 2 +
                      (q_y[ix, iy+1]*dhdy[ix, iy+1] + q_y[ix, iy]*dhdy[ix, iy]) / 2)
            potential[ix, iy] = pot
            acc += rho_w*ggrav*pot
        end

        if Sensible
            sens = (q_x[ix+1, iy]*dpwdx[ix+1, iy] + q_x[ix, iy]*dpwdx[ix, iy]) / 2 +
                   (q_y[ix, iy+1]*dpwdy[ix, iy+1] + q_y[ix, iy]*dpwdy[ix, iy]) / 2
            sensible[ix, iy] = sens
            acc += ct*cw*rho_w*sens
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
parameters (see [`MeltTerms`](@ref)). Also refreshes `s.shear`/`s.potential`/`s.sensible` as a side
effect for any term that's on (needed by [`compute_shear!`](@ref) etc. for standalone/diagnostic
use), computed via its own fused kernel rather than by calling those three functions (one launch
instead of four).

Dispatches on `sim.mt` (decided once in `Simulation`'s constructor from `p.mdot_includes_G`/
`p.mdot_includes_frictional`/`p.mdot_includes_potential`/`p.mdot_includes_sensible`/
`p.mdot_includes_qT`): a term whose flag is `false` never touches its own inputs/output field at
all (e.g. `s.sensible`/`s.dpwdx`/`s.dpwdy` when `Sensible == false`), rather than computing it and
multiplying by a zero prefactor.
"""
function compute_mdot!(s::State, p::ModelParameters, mt::MeltTerms)
    @parallel compute_mdot_kernel!(s.mdot, s.shear, s.potential, s.sensible, s.G, s.q_T, s.ub_x, s.taub_x, s.ub_y, s.taub_y, s.q_x, s.dhdx, s.q_y, s.dhdy, s.dpwdx, s.dpwdy, 1/p.L, p.rho_w, p.g, p.ct, p.cw, mt)
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
        D_x[ix, iy] = (b_x[ix, iy] / rho_i) * Linv * acc
    end
    if iy > 1 && iy < size(D_y, 2) && ix <= size(D_y, 1)
        acc = zero(eltype(D_y))
        if Potential
            acc -= rho_w * ggrav * q_y[ix, iy] * dhdy[ix, iy]
        end
        if Sensible
            acc += ct * cw * rho_w * q_y[ix, iy] * dpwdy[ix, iy]
        end
        D_y[ix, iy] = (b_y[ix, iy] / rho_i) * Linv * acc
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
