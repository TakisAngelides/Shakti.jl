"""
$(TYPEDSIGNATURES)

How `taub` (basal shear stress, feeding [`compute_mdot!`](@ref)'s frictional-heating term) is
obtained -- multiple dispatch on the concrete subtype ([`RegularizedCoulombSlidingLaw`](@ref),
[`LinearSlidingLaw`](@ref), [`PrescribedSlidingLaw`](@ref)), decided once per `Simulation` (same
idiom as [`AbstractSensibleHeatScheme`](@ref), `melt_rate.jl`).

# Notes

The original SHAKTI paper (Sommers et al. 2018) never actually specifies this -- `taub` is listed
in its Table 1 as a state variable with no formula, since it normally comes from a coupled
ice-dynamics solve; Table 2's constants have no friction coefficient at all. So this is a real
modeling choice, not just an internal solver detail; keeping it explicit and swappable rather
than silently baking in one assumption is the point of this abstraction.
"""
abstract type AbstractSlidingLaw end

"""
$(TYPEDSIGNATURES)

Regularized-Coulomb sliding law: `taub -> C*N` as `ub/N^n*lambda -> infinity` (hard Coulomb
limit), but stays finite (unlike a bare Coulomb law) as `ub -> 0`. Recomputed from `N`/`ub` every
Picard iteration (see [`compute_taub_x!`](@ref) etc. below).

# Notes

`n`/`inv_n` are read from `p.n_exp`/`p.inv_n_exp`, canonicalized once at `ModelParameters`
construction (see `model_parameters.jl`'s `pow`/`canonical_exponent` note), not per call or per
grid cell.
"""
struct RegularizedCoulombSlidingLaw{F <: AbstractFloat} <: AbstractSlidingLaw
    C::F # Coulomb friction coefficient
end

"""
$(TYPEDSIGNATURES)

`taub_x`/`taub_y` prescribed once ([`initialize_taub!`](@ref), from whatever formula or data the
caller supplies -- e.g. a driving-stress balance `taub = rho_i*g*H*(surface slope)`, the standard
parameter-free choice for synthetic slab/margin setups) and left unchanged thereafter:
[`compute_taub_x!`](@ref) etc. are no-ops for this law.
"""
struct PrescribedSlidingLaw <: AbstractSlidingLaw end

# Linear (viscous) sliding law: taub = C^2*N*u_b (Sommers and others 2023,
# Table 1). Written there as a scalar equation taub = C^2*N*|u_b| -- meant
# as "magnitude, direction opposite to u_b", the same magnitude/direction
# split RegularizedCoulombSlidingLaw's kernels use below -- but the |u_b| in
# that magnitude and the 1/|u_b| from normalizing direction cancel exactly,
# leaving a plain vector equation linear in u_b with no magnitude/direction
# split (and no ub->0 guard) needed at all.
#
# C is given at cell centers (Nx, Ny) -- e.g. an inverted per-vertex/per-cell
# drag field like Sommers and others (2023)'s friction_coefficient_Nfinal.mat
# -- or as a uniform scalar; either way it's static input data, unlike N/ub
# which genuinely change every Picard iteration, so it's staggered onto
# faces and squared once here (LinearSlidingLaw(grid, C)) rather than redone
# in the hot path.
struct LinearSlidingLaw{A <: AbstractArray} <: AbstractSlidingLaw
    Cx2::A # C^2 staggered onto x-faces (Nx+1, Ny)
    Cy2::A # C^2 staggered onto y-faces (Nx, Ny+1)
end

"""
$(TYPEDSIGNATURES)

Builds a [`LinearSlidingLaw`](@ref) on grid `g` from a drag coefficient `C` given at cell centers
(an `(nx, ny)` array, e.g. an inverted per-cell field) or as a uniform scalar, staggering it onto
faces and squaring it once here rather than in the per-iteration hot path.
"""
function LinearSlidingLaw(g::Grid, C)
    nx, ny = g.nx, g.ny
    F = eltype(g.x)
    C_cc = C isa AbstractArray ? F.(C) : fill(F(C), nx, ny) # promote a scalar to a uniform field so the staggering below is one code path either way

    # Edge faces take the value of the center, otherwise we use the arithmetic mean of the two centers next to a face
    Cx = zeros(F, nx + 1, ny)
    Cx[1, :]    .= C_cc[1, :]
    Cx[nx+1, :] .= C_cc[nx, :]
    Cx[2:nx, :] .= (C_cc[1:nx-1, :] .+ C_cc[2:nx, :]) ./ 2

    Cy = zeros(F, nx, ny + 1)
    Cy[:, 1]    .= C_cc[:, 1]
    Cy[:, ny+1] .= C_cc[:, ny]
    Cy[:, 2:ny] .= (C_cc[:, 1:ny-1] .+ C_cc[:, 2:ny]) ./ 2

    return LinearSlidingLaw(Data.Array(Cx .^ 2), Data.Array(Cy .^ 2))
end

"""
$(TYPEDSIGNATURES)

Sets up `state.taub_x`/`state.taub_y` for `sl` before the Picard loop starts. A no-op for
[`RegularizedCoulombSlidingLaw`](@ref)/[`LinearSlidingLaw`](@ref) (`taub` is instead recomputed
every Picard iteration from `N`/`ub`, see [`compute_taub_x!`](@ref) etc. below); copies `taub_x`/
`taub_y` into `state` once for [`PrescribedSlidingLaw`](@ref) (see its own docstring).
"""
initialize_taub!(::RegularizedCoulombSlidingLaw, state::State, taub_x::AbstractArray, taub_y::AbstractArray) = state
initialize_taub!(::LinearSlidingLaw, state::State, taub_x::AbstractArray, taub_y::AbstractArray) = state # recomputed every Picard iteration below, same as RegularizedCoulombSlidingLaw

function initialize_taub!(::PrescribedSlidingLaw, state::State, taub_x::AbstractArray, taub_y::AbstractArray)
    state.taub_x .= taub_x
    state.taub_y .= taub_y
    return state
end

"""
$(TYPEDSIGNATURES)

No-op for [`PrescribedSlidingLaw`](@ref): `taub_x`/`taub_y` were already set once by
[`initialize_taub!`](@ref) and never change.
"""
compute_taub_x!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s

"""
$(TYPEDSIGNATURES)

No-op for [`PrescribedSlidingLaw`](@ref), the y-face counterpart of the `compute_taub_x!` method
above.
"""
compute_taub_y!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s

"""
$(TYPEDSIGNATURES)

No-op for [`PrescribedSlidingLaw`](@ref), the fused `compute_taub_x!` + `compute_taub_y!`
counterpart above.
"""
compute_taub_xy!(s::State, p::ModelParameters, ::PrescribedSlidingLaw) = s

# taub_x/taub_y regularize on the *joint* sliding speed abs_ub = ‖v_b‖ (not
# the signed component ub_x/ub_y), then restore direction by projecting onto
# ub_x/abs_v (ub_y/abs_v) -- see Kazmierczak et al. 2024 Eq. 1 (Joughin et
# al. 2019's regularized Coulomb law) for the same vectorization, though this
# keeps Shakti's own N^n*lambda velocity scale rather than their constant v0.
# Using the signed component directly broke down at the instant a
# real (not purely +x) velocity field was used: the power-law ratio can go
# negative, and Julia's `^` throws a DomainError for a negative real base
# with the fractional exponent inv_n. ‖v_b‖ >= 0 always, so this can't happen
# here; the abs_v > 0 guard below only covers the genuine 0/0 case (both
# components vanish at a face).
#
# abs_ub is reused rather than re-derived locally from ub_x/ub_y: it's
# already the correct cell-centered ‖v_b‖ (compute_abs_ub_kernel!), and valid
# by construction here since it's computed right before these calls (see
# initial_conditions.jl and elliptic_solver.jl). This currently holds for an
# entire run because ub_x/ub_y are set once in set_initial_conditions! and
# never reassigned afterward -- NOTE: once ice-velocity coupling starts
# updating ub_x/ub_y mid-run (every timestep, or every few timesteps under a
# frozen-ice-velocity assumption), compute_abs_ub! must be re-invoked
# whenever ub_x/ub_y change, before the next compute_taub_x!/_y!/_xy! call,
# or abs_ub (and therefore taub) will silently go stale.
@parallel_indices (ix, iy) function compute_taub_x_kernel!(taub_x, N, ub_x, abs_ub, lambda, C, n, inv_n)
    nx1 = size(taub_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(taub_x, 2)
        if ix == 1
            Nf = N[1, iy]
            abs_v = abs_ub[1, iy]
            taub_x[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lambda[1, iy]), inv_n) * (ub_x[1, iy] / abs_v) : zero(Nf)
        elseif ix == nx1
            Nf = N[nx1-1, iy]
            abs_v = abs_ub[nx1-1, iy]
            taub_x[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lambda[nx1-1, iy]), inv_n) * (ub_x[nx1, iy] / abs_v) : zero(Nf)
        else
            Nf = (N[ix, iy] + N[ix-1, iy]) / 2
            lf = (lambda[ix, iy] + lambda[ix-1, iy]) / 2
            abs_v = (abs_ub[ix, iy] + abs_ub[ix-1, iy]) / 2
            taub_x[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lf), inv_n) * (ub_x[ix, iy] / abs_v) : zero(Nf)
        end
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.taub_x` under [`RegularizedCoulombSlidingLaw`](@ref) from the current `s.N`/`s.ub_x`.
"""
compute_taub_x!(s::State, p::ModelParameters, sl::RegularizedCoulombSlidingLaw) = (@parallel compute_taub_x_kernel!(s.taub_x, s.N, s.ub_x, s.abs_ub, s.lambda, sl.C, p.n_exp, p.inv_n_exp); s)

@parallel_indices (ix, iy) function compute_taub_y_kernel!(taub_y, N, ub_y, abs_ub, lambda, C, n, inv_n)
    ny1 = size(taub_y, 2) # ny + 1
    if ix <= size(taub_y, 1) && iy <= ny1
        if iy == 1
            Nf = N[ix, 1]
            abs_v = abs_ub[ix, 1]
            taub_y[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lambda[ix, 1]), inv_n) * (ub_y[ix, 1] / abs_v) : zero(Nf)
        elseif iy == ny1
            Nf = N[ix, ny1-1]
            abs_v = abs_ub[ix, ny1-1]
            taub_y[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lambda[ix, ny1-1]), inv_n) * (ub_y[ix, ny1] / abs_v) : zero(Nf)
        else
            Nf = (N[ix, iy] + N[ix, iy-1]) / 2
            lf = (lambda[ix, iy] + lambda[ix, iy-1]) / 2
            abs_v = (abs_ub[ix, iy] + abs_ub[ix, iy-1]) / 2
            taub_y[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lf), inv_n) * (ub_y[ix, iy] / abs_v) : zero(Nf)
        end
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.taub_y` under [`RegularizedCoulombSlidingLaw`](@ref), the y-face counterpart of the
`compute_taub_x!` method above.
"""
compute_taub_y!(s::State, p::ModelParameters, sl::RegularizedCoulombSlidingLaw) = (@parallel compute_taub_y_kernel!(s.taub_y, s.N, s.ub_y, s.abs_ub, s.lambda, sl.C, p.n_exp, p.inv_n_exp); s)

# Fused hot-path version: one launch instead of two (see field_gradients.jl's
# compute_dhdxy! for why passing both differently-shaped face arrays as
# arguments makes ParallelStencil infer the right union launch range).
@parallel_indices (ix, iy) function compute_taub_xy_kernel!(taub_x, taub_y, N, ub_x, ub_y, abs_ub, lambda, C, n, inv_n)
    nx1 = size(taub_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(taub_x, 2)
        if ix == 1
            Nf = N[1, iy]
            abs_v = abs_ub[1, iy]
            taub_x[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lambda[1, iy]), inv_n) * (ub_x[1, iy] / abs_v) : zero(Nf)
        elseif ix == nx1
            Nf = N[nx1-1, iy]
            abs_v = abs_ub[nx1-1, iy]
            taub_x[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lambda[nx1-1, iy]), inv_n) * (ub_x[nx1, iy] / abs_v) : zero(Nf)
        else
            Nf = (N[ix, iy] + N[ix-1, iy]) / 2
            lf = (lambda[ix, iy] + lambda[ix-1, iy]) / 2
            abs_v = (abs_ub[ix, iy] + abs_ub[ix-1, iy]) / 2
            taub_x[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lf), inv_n) * (ub_x[ix, iy] / abs_v) : zero(Nf)
        end
    end
    ny1 = size(taub_y, 2) # ny + 1
    if ix <= size(taub_y, 1) && iy <= ny1
        if iy == 1
            Nf = N[ix, 1]
            abs_v = abs_ub[ix, 1]
            taub_y[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lambda[ix, 1]), inv_n) * (ub_y[ix, 1] / abs_v) : zero(Nf)
        elseif iy == ny1
            Nf = N[ix, ny1-1]
            abs_v = abs_ub[ix, ny1-1]
            taub_y[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lambda[ix, ny1-1]), inv_n) * (ub_y[ix, ny1] / abs_v) : zero(Nf)
        else
            Nf = (N[ix, iy] + N[ix, iy-1]) / 2
            lf = (lambda[ix, iy] + lambda[ix, iy-1]) / 2
            abs_v = (abs_ub[ix, iy] + abs_ub[ix, iy-1]) / 2
            taub_y[ix, iy] = abs_v > 0 ? Nf * C * pow(abs_v / (abs_v + pow(abs(Nf), n) * lf), inv_n) * (ub_y[ix, iy] / abs_v) : zero(Nf)
        end
    end
    return
end
"""
$(TYPEDSIGNATURES)

Fused version of `compute_taub_x!` + `compute_taub_y!` under [`RegularizedCoulombSlidingLaw`](@ref):
one `@parallel` launch instead of two.
"""
compute_taub_xy!(s::State, p::ModelParameters, sl::RegularizedCoulombSlidingLaw) = (@parallel compute_taub_xy_kernel!(s.taub_x, s.taub_y, s.N, s.ub_x, s.ub_y, s.abs_ub, s.lambda, sl.C, p.n_exp, p.inv_n_exp); s)

# taub = C^2*N*u_b (LinearSlidingLaw, see its struct docstring above): N
# staggered onto the face with the same boundary-duplicate/interior-average
# convention as RegularizedCoulombSlidingLaw's kernels, Cx2/Cy2 already
# staggered+squared once at construction. No abs_ub/lambda/n/inv_n needed --
# this law is linear straight through ub=0, unlike the regularized-Coulomb
# power-law ratio.
@parallel_indices (ix, iy) function compute_taub_x_linear_kernel!(taub_x, N, ub_x, Cx2)
    nx1 = size(taub_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(taub_x, 2)
        Nf = ix == 1 ? N[1, iy] : ix == nx1 ? N[nx1-1, iy] : (N[ix, iy] + N[ix-1, iy]) / 2
        taub_x[ix, iy] = Cx2[ix, iy] * Nf * ub_x[ix, iy]
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.taub_x` under [`LinearSlidingLaw`](@ref) from the current `s.N`/`s.ub_x`.
"""
compute_taub_x!(s::State, p::ModelParameters, sl::LinearSlidingLaw) = (@parallel compute_taub_x_linear_kernel!(s.taub_x, s.N, s.ub_x, sl.Cx2); s)

@parallel_indices (ix, iy) function compute_taub_y_linear_kernel!(taub_y, N, ub_y, Cy2)
    ny1 = size(taub_y, 2) # ny + 1
    if ix <= size(taub_y, 1) && iy <= ny1
        Nf = iy == 1 ? N[ix, 1] : iy == ny1 ? N[ix, ny1-1] : (N[ix, iy] + N[ix, iy-1]) / 2
        taub_y[ix, iy] = Cy2[ix, iy] * Nf * ub_y[ix, iy]
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.taub_y` under [`LinearSlidingLaw`](@ref), the y-face counterpart of the
`compute_taub_x!` method above.
"""
compute_taub_y!(s::State, p::ModelParameters, sl::LinearSlidingLaw) = (@parallel compute_taub_y_linear_kernel!(s.taub_y, s.N, s.ub_y, sl.Cy2); s)

# Fused hot-path version, same rationale as compute_taub_xy_kernel! above.
@parallel_indices (ix, iy) function compute_taub_xy_linear_kernel!(taub_x, taub_y, N, ub_x, ub_y, Cx2, Cy2)
    nx1 = size(taub_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(taub_x, 2)
        Nf = ix == 1 ? N[1, iy] : ix == nx1 ? N[nx1-1, iy] : (N[ix, iy] + N[ix-1, iy]) / 2
        taub_x[ix, iy] = Cx2[ix, iy] * Nf * ub_x[ix, iy]
    end
    ny1 = size(taub_y, 2) # ny + 1
    if ix <= size(taub_y, 1) && iy <= ny1
        Nf = iy == 1 ? N[ix, 1] : iy == ny1 ? N[ix, ny1-1] : (N[ix, iy] + N[ix, iy-1]) / 2
        taub_y[ix, iy] = Cy2[ix, iy] * Nf * ub_y[ix, iy]
    end
    return
end
"""
$(TYPEDSIGNATURES)

Fused version of `compute_taub_x!` + `compute_taub_y!` under [`LinearSlidingLaw`](@ref): one
`@parallel` launch instead of two.
"""
compute_taub_xy!(s::State, p::ModelParameters, sl::LinearSlidingLaw) = (@parallel compute_taub_xy_linear_kernel!(s.taub_x, s.taub_y, s.N, s.ub_x, s.ub_y, sl.Cx2, sl.Cy2); s)
