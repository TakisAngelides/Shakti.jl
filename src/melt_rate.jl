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

# Regularized-Coulomb sliding law: taub -> C*N as ub/N^n*lambda -> infinity
# (hard Coulomb limit), but stays finite (unlike a bare Coulomb law) as
# ub -> 0. n/inv_n read from p.n_exp/p.inv_n_exp, canonicalized once at
# ModelParameters construction (see model_parameters.jl's pow/
# canonical_exponent note), not per call or per grid cell.
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
compute_taub_x!(s::State, p::ModelParameters) = (@parallel compute_taub_x_kernel!(s.taub_x, s.N, s.ub_x, s.lambda, p.C, p.n_exp, p.inv_n_exp); s)

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
compute_taub_y!(s::State, p::ModelParameters) = (@parallel compute_taub_y_kernel!(s.taub_y, s.N, s.ub_y, s.lambda, p.C, p.n_exp, p.inv_n_exp); s)

# Melt rate = geothermal flux + frictional (sliding) heating + potential
# energy released by water flowing downgradient + sensible heat exchanged as
# water moves to regions of different pressure melting point, all divided by
# the latent heat of fusion L. The three heat-source terms are broken out
# into their own kernels below, writing into preallocated State fields
# (s.shear/s.potential/s.sensible) rather than local per-cell scalars, so
# compute_mdot! can be called every Picard iteration without allocating.

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

@parallel_indices (ix, iy) function compute_mdot_kernel!(mdot, G, shear, potential, sensible, Linv, rho_w, ggrav, ct, cw)
    if ix <= size(mdot, 1) && iy <= size(mdot, 2)
        mdot[ix, iy] = Linv * (G[ix, iy] + shear[ix, iy] + rho_w*ggrav*potential[ix, iy] + ct*cw*rho_w*sensible[ix, iy])
    end
    return
end

@parallel_indices (ix, iy) function compute_mdot_kernel!(mdot, G, shear, potential, Linv, rho_w, ggrav)
    if ix <= size(mdot, 1) && iy <= size(mdot, 2)
        mdot[ix, iy] = Linv * (G[ix, iy] + shear[ix, iy] + rho_w*ggrav*potential[ix, iy])
    end
    return
end

# sim.shs (WithSensibleHeat()/NoSensibleHeat(), decided once in Simulation's
# constructor from p.ct/p.cw -- see simulation.jl) picks which method
# compiles in: the NoSensibleHeat path never launches compute_sensible! or
# touches s.sensible at all, rather than computing it and multiplying by a
# zero ct*cw prefactor.
function compute_mdot!(s::State, p::ModelParameters, ::WithSensibleHeat)
    compute_shear!(s)
    compute_potential!(s)
    compute_sensible!(s)
    @parallel compute_mdot_kernel!(s.mdot, s.G, s.shear, s.potential, s.sensible, 1/p.L, p.rho_w, p.g, p.ct, p.cw)
    return s
end

function compute_mdot!(s::State, p::ModelParameters, ::NoSensibleHeat)
    compute_shear!(s)
    compute_potential!(s)
    @parallel compute_mdot_kernel!(s.mdot, s.G, s.shear, s.potential, 1/p.L, p.rho_w, p.g)
    return s
end