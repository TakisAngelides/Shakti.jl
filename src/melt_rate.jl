# Whether compute_mdot! includes the sensible-heat term. Decided ONCE in
# Simulation's constructor (from p.ct/p.cw, see simulation.jl) rather than
# every call, so turning the term off skips compute_sensible!'s kernel launch
# and array traffic entirely instead of just multiplying its contribution by
# zero -- same "dispatch on a type decided once outside the hot loop" idiom
# as canonical_exponent/pow (model_parameters.jl) and the head/gap schemes
# (simulation.jl). Defined here, next to their sole consumer (compute_mdot!
# below), rather than in simulation.jl: elliptic_solver.jl (included before
# simulation.jl for its own PicardSolver/Simulation ordering reasons) needs
# AbstractSensibleHeatScheme to type-annotate shs, so this file is included
# before elliptic_solver.jl too.
abstract type AbstractSensibleHeatScheme end

struct WithSensibleHeat <: AbstractSensibleHeatScheme end
struct NoSensibleHeat   <: AbstractSensibleHeatScheme end

# How taub (basal shear stress, feeding compute_mdot!'s frictional-heating
# term below) is obtained. The original SHAKTI paper (Sommers et al. 2018)
# never actually specifies this -- taub is listed in its Table 1 as a state
# variable with no formula, since it normally comes from a coupled ice-
# dynamics solve; Table 2's constants have no friction coefficient at all.
# So this is a real modeling choice, not just an internal solver detail --
# multiple dispatch on sl::AbstractSlidingLaw, decided once per Simulation
# (same idiom as AbstractSensibleHeatScheme above), keeps that choice
# explicit and swappable instead of silently baking in one assumption.
abstract type AbstractSlidingLaw end

# Regularized-Coulomb sliding law: taub -> C*N as ub/N^n*lambda -> infinity
# (hard Coulomb limit), but stays finite (unlike a bare Coulomb law) as
# ub -> 0. Recomputed from N/ub every Picard iteration (see compute_taub_x!
# etc. below). n/inv_n read from p.n_exp/p.inv_n_exp, canonicalized once at
# ModelParameters construction (see model_parameters.jl's pow/
# canonical_exponent note), not per call or per grid cell.
struct RegularizedCoulombSlidingLaw{F <: AbstractFloat} <: AbstractSlidingLaw
    C::F # Coulomb friction coefficient
end

# taub_x/taub_y prescribed once (initialize_taub!, from whatever formula or
# data the caller supplies -- e.g. a driving-stress balance
# taub = rho_i*g*H*(surface slope), the standard parameter-free choice for
# synthetic slab/margin setups) and left unchanged thereafter: compute_taub_x!
# etc. below are no-ops for this law.
struct PrescribedSlidingLaw <: AbstractSlidingLaw end

initialize_taub!(::RegularizedCoulombSlidingLaw, state::State, taub_x::AbstractArray, taub_y::AbstractArray) = state

function initialize_taub!(::PrescribedSlidingLaw, state::State, taub_x::AbstractArray, taub_y::AbstractArray)
    state.taub_x .= taub_x
    state.taub_y .= taub_y
    return state
end

compute_taub_x!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s
compute_taub_y!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s
compute_taub_xy!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s

@parallel_indices (ix, iy) function compute_taub_x_kernel!(taub_x, N, ub_x, lambda, C, n, inv_n)
    nx1 = size(taub_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(taub_x, 2)
        if ix == 1
            Nf = N[1, iy]
            taub_x[ix, iy] = Nf * C * pow(ub_x[1, iy] / (ub_x[1, iy] + pow(abs(Nf), n) * lambda[1, iy]), inv_n)
        elseif ix == nx1
            Nf = N[nx1-1, iy]
            taub_x[ix, iy] = Nf * C * pow(ub_x[nx1, iy] / (ub_x[nx1, iy] + pow(abs(Nf), n) * lambda[nx1-1, iy]), inv_n)
        else
            Nf = (N[ix, iy] + N[ix-1, iy]) / 2
            lf = (lambda[ix, iy] + lambda[ix-1, iy]) / 2
            taub_x[ix, iy] = Nf * C * pow(ub_x[ix, iy] / (ub_x[ix, iy] + pow(abs(Nf), n) * lf), inv_n)
        end
    end
    return
end
compute_taub_x!(s::State, p::ModelParameters, sl::RegularizedCoulombSlidingLaw) = (@parallel compute_taub_x_kernel!(s.taub_x, s.N, s.ub_x, s.lambda, sl.C, p.n_exp, p.inv_n_exp); s)

@parallel_indices (ix, iy) function compute_taub_y_kernel!(taub_y, N, ub_y, lambda, C, n, inv_n)
    ny1 = size(taub_y, 2) # ny + 1
    if ix <= size(taub_y, 1) && iy <= ny1
        if iy == 1
            Nf = N[ix, 1]
            taub_y[ix, iy] = Nf * C * pow(ub_y[ix, 1] / (ub_y[ix, 1] + pow(abs(Nf), n) * lambda[ix, 1]), inv_n)
        elseif iy == ny1
            Nf = N[ix, ny1-1]
            taub_y[ix, iy] = Nf * C * pow(ub_y[ix, ny1] / (ub_y[ix, ny1] + pow(abs(Nf), n) * lambda[ix, ny1-1]), inv_n)
        else
            Nf = (N[ix, iy] + N[ix, iy-1]) / 2
            lf = (lambda[ix, iy] + lambda[ix, iy-1]) / 2
            taub_y[ix, iy] = Nf * C * pow(ub_y[ix, iy] / (ub_y[ix, iy] + pow(abs(Nf), n) * lf), inv_n)
        end
    end
    return
end
compute_taub_y!(s::State, p::ModelParameters, sl::RegularizedCoulombSlidingLaw) = (@parallel compute_taub_y_kernel!(s.taub_y, s.N, s.ub_y, s.lambda, sl.C, p.n_exp, p.inv_n_exp); s)

# Fused hot-path version: one launch instead of two (see field_gradients.jl's
# compute_dhdxy! for why passing both differently-shaped face arrays as
# arguments makes ParallelStencil infer the right union launch range).
@parallel_indices (ix, iy) function compute_taub_xy_kernel!(taub_x, taub_y, N, ub_x, ub_y, lambda, C, n, inv_n)
    nx1 = size(taub_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(taub_x, 2)
        if ix == 1
            Nf = N[1, iy]
            taub_x[ix, iy] = Nf * C * pow(ub_x[1, iy] / (ub_x[1, iy] + pow(abs(Nf), n) * lambda[1, iy]), inv_n)
        elseif ix == nx1
            Nf = N[nx1-1, iy]
            taub_x[ix, iy] = Nf * C * pow(ub_x[nx1, iy] / (ub_x[nx1, iy] + pow(abs(Nf), n) * lambda[nx1-1, iy]), inv_n)
        else
            Nf = (N[ix, iy] + N[ix-1, iy]) / 2
            lf = (lambda[ix, iy] + lambda[ix-1, iy]) / 2
            taub_x[ix, iy] = Nf * C * pow(ub_x[ix, iy] / (ub_x[ix, iy] + pow(abs(Nf), n) * lf), inv_n)
        end
    end
    ny1 = size(taub_y, 2) # ny + 1
    if ix <= size(taub_y, 1) && iy <= ny1
        if iy == 1
            Nf = N[ix, 1]
            taub_y[ix, iy] = Nf * C * pow(ub_y[ix, 1] / (ub_y[ix, 1] + pow(abs(Nf), n) * lambda[ix, 1]), inv_n)
        elseif iy == ny1
            Nf = N[ix, ny1-1]
            taub_y[ix, iy] = Nf * C * pow(ub_y[ix, ny1] / (ub_y[ix, ny1] + pow(abs(Nf), n) * lambda[ix, ny1-1]), inv_n)
        else
            Nf = (N[ix, iy] + N[ix, iy-1]) / 2
            lf = (lambda[ix, iy] + lambda[ix, iy-1]) / 2
            taub_y[ix, iy] = Nf * C * pow(ub_y[ix, iy] / (ub_y[ix, iy] + pow(abs(Nf), n) * lf), inv_n)
        end
    end
    return
end
compute_taub_xy!(s::State, p::ModelParameters, sl::RegularizedCoulombSlidingLaw) = (@parallel compute_taub_xy_kernel!(s.taub_x, s.taub_y, s.N, s.ub_x, s.ub_y, s.lambda, sl.C, p.n_exp, p.inv_n_exp); s)

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
compute_shear!(s::State) = (@parallel compute_shear_kernel!(s.shear, s.ub_x, s.taub_x, s.ub_y, s.taub_y); s)

@parallel_indices (ix, iy) function compute_potential_kernel!(potential, q_x, dhdx, q_y, dhdy)
    if ix <= size(potential, 1) && iy <= size(potential, 2)
        potential[ix, iy] = abs((q_x[ix+1, iy]*dhdx[ix+1, iy] + q_x[ix, iy]*dhdx[ix, iy]) / 2 +
                                (q_y[ix, iy+1]*dhdy[ix, iy+1] + q_y[ix, iy]*dhdy[ix, iy]) / 2)
    end
    return
end
compute_potential!(s::State) = (@parallel compute_potential_kernel!(s.potential, s.q_x, s.dhdx, s.q_y, s.dhdy); s)

# Requires dpwdx/dpwdy (computed above) already current.
@parallel_indices (ix, iy) function compute_sensible_kernel!(sensible, q_x, dpwdx, q_y, dpwdy)
    if ix <= size(sensible, 1) && iy <= size(sensible, 2)
        sensible[ix, iy] = (q_x[ix+1, iy]*dpwdx[ix+1, iy] + q_x[ix, iy]*dpwdx[ix, iy]) / 2 +
                            (q_y[ix, iy+1]*dpwdy[ix, iy+1] + q_y[ix, iy]*dpwdy[ix, iy]) / 2
    end
    return
end
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

# sim.shs (WithSensibleHeat()/NoSensibleHeat(), decided once in Simulation's
# constructor from p.ct/p.cw -- see simulation.jl) picks which method
# compiles in: the NoSensibleHeat path never touches s.sensible or dpwdx/dpwdy
# at all, rather than computing sensible and multiplying by a zero ct*cw
# prefactor.
function compute_mdot!(s::State, p::ModelParameters, ::WithSensibleHeat)
    @parallel compute_mdot_kernel!(s.mdot, s.shear, s.potential, s.sensible, s.G, s.ub_x, s.taub_x, s.ub_y, s.taub_y, s.q_x, s.dhdx, s.q_y, s.dhdy, s.dpwdx, s.dpwdy, 1/p.L, p.rho_w, p.g, p.ct, p.cw)
    return s
end

function compute_mdot!(s::State, p::ModelParameters, ::NoSensibleHeat)
    @parallel compute_mdot_kernel!(s.mdot, s.shear, s.potential, s.G, s.ub_x, s.taub_x, s.ub_y, s.taub_y, s.q_x, s.dhdx, s.q_y, s.dhdy, 1/p.L, p.rho_w, p.g)
    return s
end