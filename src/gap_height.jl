# =============================================================================
# Gap height (b) evolution
# =============================================================================
# Runs once per time step, after the Picard loop has converged on h. compute_beta!
# and compute_b_x!/compute_b_y! are then refreshed from the new b, ready to
# be read by the next time step's Picard loop (compute_q_x!/compute_q_y!
# read b_x/b_y; the next compute_b! call reads beta).
#
# This file is included before simulation.jl (same reason melt_rate.jl is
# included before elliptic_solver.jl, see its own docstring note):
# Simulation's struct definition needs AbstractOpenBySlidingScheme in scope to
# type-annotate `oss`. The Simulation-dispatching compute_b!(sim::Simulation)
# family (mirroring compute_b! dispatching on sim.gs) therefore lives in
# run.jl instead, next to step_b! -- their sole caller -- rather than here.

"""
$(TYPEDSIGNATURES)

Whether `compute_beta!` includes the opening-by-sliding term -- multiple dispatch on the concrete
subtype ([`WithOpenBySliding`](@ref)/[`NoOpenBySliding`](@ref)), decided ONCE in `Simulation`'s
constructor (from `p.br`, see `simulation.jl`) rather than every call, so turning it off skips
`compute_beta_kernel!`'s kernel launch entirely instead of launching it to compute a term that's
mathematically always zero when `p.br == 0` -- same "dispatch on a type decided once outside the
hot loop" idiom as [`AbstractSensibleHeatScheme`](@ref) (`melt_rate.jl`).
"""
abstract type AbstractOpenBySlidingScheme end

"""
$(TYPEDSIGNATURES)

Include the opening-by-sliding term (`compute_beta_kernel!`) in [`compute_beta!`](@ref).
"""
struct WithOpenBySliding <: AbstractOpenBySlidingScheme end

"""
$(TYPEDSIGNATURES)

Skip the opening-by-sliding term in [`compute_beta!`](@ref) entirely (not just compute a term
that's identically zero): `s.beta` is left untouched, which is correct since `p.br == 0` is what
selects this scheme in the first place, and `max(0, (0 - b)/lr) == 0` for any `b >= 0` anyway.
"""
struct NoOpenBySliding <: AbstractOpenBySlidingScheme end

# Opening rate of the gap by sliding over bedrock bumps of height br and
# spacing lr (Rothlisberger-style cavity opening): positive only while the
# gap is still smaller than the bump height, zero once b has grown past br.
@parallel_indices (ix, iy) function compute_beta_kernel!(beta, b, br, lr)
    if ix <= size(beta, 1) && iy <= size(beta, 2)
        beta[ix, iy] = max(zero(br), (br - b[ix, iy]) / lr)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.beta` (opening rate of the gap by sliding over bedrock bumps of height `p.br` and
spacing `p.lr`, Röthlisberger-style cavity opening): positive only while the gap is still smaller
than the bump height, zero once `b` has grown past `br`.

Dispatches on `oss` (decided once, from `p.br`, see [`AbstractOpenBySlidingScheme`](@ref)): the
[`NoOpenBySliding`](@ref) method skips the kernel launch entirely, since `p.br == 0` makes the term
identically zero mathematically.
"""
compute_beta!(s::State, p::ModelParameters, ::WithOpenBySliding) = (@parallel compute_beta_kernel!(s.beta, s.b, p.br, p.lr); s)

"""
$(TYPEDSIGNATURES)

`compute_beta!` without the opening-by-sliding term -- see the `WithOpenBySliding` method's
docstring above.
"""
compute_beta!(s::State, p::ModelParameters, ::NoOpenBySliding) = s

@parallel_indices (ix, iy) function compute_b_x_kernel!(b_x, b)
    nx1 = size(b_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(b_x, 2)
        if ix == 1 # boundary face takes the value of the nearest center value
            b_x[ix, iy] = b[1, iy]
        elseif ix == nx1 # boundary face takes the value of the nearest center value
            b_x[ix, iy] = b[nx1-1, iy]
        else # face value is taken to be the arithmetic average of the center values touching that face
            b_x[ix, iy] = (b[ix, iy] + b[ix-1, iy]) / 2
        end
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.b_x` by staggering `s.b` onto x-faces (boundary faces duplicate the nearest cell,
interior faces average their two neighbours).
"""
compute_b_x!(s::State) = (@parallel compute_b_x_kernel!(s.b_x, s.b); s)

@parallel_indices (ix, iy) function compute_b_y_kernel!(b_y, b)
    ny1 = size(b_y, 2) # ny + 1
    if ix <= size(b_y, 1) && iy <= ny1
        if iy == 1
            b_y[ix, iy] = b[ix, 1]
        elseif iy == ny1
            b_y[ix, iy] = b[ix, ny1-1]
        else
            b_y[ix, iy] = (b[ix, iy] + b[ix, iy-1]) / 2
        end
    end
    return
end
"""
$(TYPEDSIGNATURES)

Updates `s.b_y` by staggering `s.b` onto y-faces (boundary faces duplicate the nearest cell,
interior faces average their two neighbours).
"""
compute_b_y!(s::State) = (@parallel compute_b_y_kernel!(s.b_y, s.b); s)

@parallel_indices (ix, iy) function compute_b_implicit_kernel!(b, mask, mdot, beta, abs_ub, A_visc, N, rho_i, n_minus_1, dt, b_min, b_max)
    if ix <= size(b, 1) && iy <= size(b, 2) && mask[ix, iy] == GROUNDED # we only evolve the water thickness if the cell has grounded ice
        b[ix, iy] = clamp( # b_min/b_max bound the water thickness for numerical stability -- b_max specifically guards against a runaway b->K->q->mdot->b feedback (creep closure, which depends on N, vanishes as N->0, so nothing bounds b from above there without this; see ModelParameters' own docstring)
            (b[ix, iy] + dt * (mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy])) /
            (1 + dt * A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]),
            b_min, b_max)
    end
    return
end

@parallel_indices (ix, iy) function compute_b_explicit_kernel!(b, mask, mdot, beta, abs_ub, A_visc, N, rho_i, n_minus_1, dt, b_min, b_max)
    if ix <= size(b, 1) && iy <= size(b, 2) && mask[ix, iy] == GROUNDED
        b[ix, iy] = clamp(
            b[ix, iy] + dt * (mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy] -
                A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * b[ix, iy]),
            b_min, b_max)
    end
    return
end
