"""
$(TYPEDSIGNATURES)

Whether `compute_mdot!` includes the sensible-heat term -- multiple dispatch on the concrete
subtype ([`WithSensibleHeat`](@ref)/[`NoSensibleHeat`](@ref)), decided ONCE in `Simulation`'s
constructor (from `p.ct`/`p.cw`, see `simulation.jl`) rather than every call, so turning the term
off skips `compute_sensible!`'s kernel launch and array traffic entirely instead of just
multiplying its contribution by zero -- same "dispatch on a type decided once outside the hot
loop" idiom as `canonical_exponent`/`pow` (`model_parameters.jl`) and the head/gap schemes
(`simulation.jl`).

# Notes

Defined here, next to their sole consumer ([`compute_mdot!`](@ref) below), rather than in
`simulation.jl`: `elliptic_solver.jl` (included before `simulation.jl` for its own
`PicardSolver`/`Simulation` ordering reasons) needs `AbstractSensibleHeatScheme` to type-annotate
`shs`, so this file is included before `elliptic_solver.jl` too.
"""
abstract type AbstractSensibleHeatScheme end

"""
$(TYPEDSIGNATURES)

Include the sensible-heat term (`ct*cw*rho_w*sens`) in [`compute_mdot!`](@ref).
"""
struct WithSensibleHeat <: AbstractSensibleHeatScheme end

"""
$(TYPEDSIGNATURES)

Skip the sensible-heat term in [`compute_mdot!`](@ref) entirely (not just multiply by zero).
"""
struct NoSensibleHeat   <: AbstractSensibleHeatScheme end

# Melt rate = geothermal flux + frictional (sliding) heating + potential
# energy released by water flowing downgradient + sensible heat exchanged as
# water moves to regions of different pressure melting point, all divided by
# the latent heat of fusion L. The three heat-source terms are exposed as
# their own standalone kernels below (compute_shear!/compute_potential!/
# compute_sensible!, writing into preallocated State fields s.shear/
# s.potential/s.sensible for standalone/diagnostic use, e.g. as a tracked_obs
# name -- see Simulation's tracked_obs), but compute_mdot!'s hot path (below)
# does NOT call them: it uses its own fused kernel that recomputes the same
# three terms and combines them into mdot in a single kernel launch instead
# of four, while still writing shear/potential/sensible so those fields stay
# valid every Picard iteration for anyone reading them.

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

# Fused hot-path kernel: shear/potential/sensible/mdot in one launch instead of
# four. Duplicates the per-cell math above rather than calling those kernels,
# since each is itself a separate kernel launch; still writes shear/potential/
# sensible (not just mdot) so those fields aren't left stale for anything that
# reads them after a Picard iteration.
@parallel_indices (ix, iy) function compute_mdot_kernel!(mdot, shear, potential, sensible, G, ub_x, taub_x, ub_y, taub_y, q_x, dhdx, q_y, dhdy, dpwdx, dpwdy, Linv, rho_w, ggrav, ct, cw)
    if ix <= size(mdot, 1) && iy <= size(mdot, 2)
        sh   = abs((ub_x[ix+1, iy]*taub_x[ix+1, iy] + ub_x[ix, iy]*taub_x[ix, iy]) / 2 +
                   (ub_y[ix, iy+1]*taub_y[ix, iy+1] + ub_y[ix, iy]*taub_y[ix, iy]) / 2)
        pot  = abs((q_x[ix+1, iy]*dhdx[ix+1, iy] + q_x[ix, iy]*dhdx[ix, iy]) / 2 +
                   (q_y[ix, iy+1]*dhdy[ix, iy+1] + q_y[ix, iy]*dhdy[ix, iy]) / 2)
        sens = (q_x[ix+1, iy]*dpwdx[ix+1, iy] + q_x[ix, iy]*dpwdx[ix, iy]) / 2 +
               (q_y[ix, iy+1]*dpwdy[ix, iy+1] + q_y[ix, iy]*dpwdy[ix, iy]) / 2

        shear[ix, iy]     = sh
        potential[ix, iy] = pot
        sensible[ix, iy]  = sens

        mdot[ix, iy] = Linv * (G[ix, iy] + sh + rho_w*ggrav*pot + ct*cw*rho_w*sens)
    end
    return
end

@parallel_indices (ix, iy) function compute_mdot_kernel!(mdot, shear, potential, G, ub_x, taub_x, ub_y, taub_y, q_x, dhdx, q_y, dhdy, Linv, rho_w, ggrav)
    if ix <= size(mdot, 1) && iy <= size(mdot, 2)
        sh  = abs((ub_x[ix+1, iy]*taub_x[ix+1, iy] + ub_x[ix, iy]*taub_x[ix, iy]) / 2 +
                  (ub_y[ix, iy+1]*taub_y[ix, iy+1] + ub_y[ix, iy]*taub_y[ix, iy]) / 2)
        pot = abs((q_x[ix+1, iy]*dhdx[ix+1, iy] + q_x[ix, iy]*dhdx[ix, iy]) / 2 +
                  (q_y[ix, iy+1]*dhdy[ix, iy+1] + q_y[ix, iy]*dhdy[ix, iy]) / 2)

        shear[ix, iy]     = sh
        potential[ix, iy] = pot

        mdot[ix, iy] = Linv * (G[ix, iy] + sh + rho_w*ggrav*pot)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.mdot` (subglacial melt rate) = geothermal flux + frictional (sliding) heating +
potential energy released by water flowing downgradient + sensible heat exchanged as water moves
to regions of different pressure melting point, all divided by the latent heat of fusion `p.L`.
Also refreshes `s.shear`/`s.potential`/`s.sensible` as a side effect (needed by
[`compute_shear!`](@ref) etc. for standalone/diagnostic use), computed via its own fused kernel
rather than by calling those three functions (one launch instead of four).

Dispatches on `sim.shs` (decided once in `Simulation`'s constructor from `p.ct`/`p.cw`, see
[`AbstractSensibleHeatScheme`](@ref)): the [`NoSensibleHeat`](@ref) method never touches
`s.sensible` or `s.dpwdx`/`s.dpwdy` at all, rather than computing `sensible` and multiplying by a
zero `ct*cw` prefactor.
"""
function compute_mdot!(s::State, p::ModelParameters, ::WithSensibleHeat)
    @parallel compute_mdot_kernel!(s.mdot, s.shear, s.potential, s.sensible, s.G, s.ub_x, s.taub_x, s.ub_y, s.taub_y, s.q_x, s.dhdx, s.q_y, s.dhdy, s.dpwdx, s.dpwdy, 1/p.L, p.rho_w, p.g, p.ct, p.cw)
    return s
end

"""
$(TYPEDSIGNATURES)

`compute_mdot!` without the sensible-heat term -- see the `WithSensibleHeat` method's docstring
above.
"""
function compute_mdot!(s::State, p::ModelParameters, ::NoSensibleHeat)
    @parallel compute_mdot_kernel!(s.mdot, s.shear, s.potential, s.G, s.ub_x, s.taub_x, s.ub_y, s.taub_y, s.q_x, s.dhdx, s.q_y, s.dhdy, 1/p.L, p.rho_w, p.g)
    return s
end