# =============================================================================
# Gap height (b) evolution
# =============================================================================
# Runs once per time step, after the Picard loop above has converged on h
# (and everything it drives: N, mdot, taub_x/taub_y, abs_ub). compute_beta!
# and compute_b_x!/compute_b_y! are then refreshed from the new b, ready to
# be read by the *next* time step's Picard loop (compute_q_x!/compute_q_y!
# read b_x/b_y; the next compute_b! call reads beta).

# Opening rate of the gap by sliding over bedrock bumps of height br and
# spacing lr (Rothlisberger-style cavity opening): positive only while the
# gap is still smaller than the bump height, zero once b has grown past br.
@parallel_indices (ix, iy) function compute_beta_kernel!(beta, b, br, lr)
    if ix <= size(beta, 1) && iy <= size(beta, 2)
        beta[ix, iy] = max(zero(br), (br - b[ix, iy]) / lr)
    end
    return
end
compute_beta!(s::State, p::ModelParameters) = (@parallel compute_beta_kernel!(s.beta, s.b, p.br, p.lr); s)

@parallel_indices (ix, iy) function compute_b_x_kernel!(b_x, b)
    nx1 = size(b_x, 1) # nx + 1
    if ix <= nx1 && iy <= size(b_x, 2)
        if ix == 1
            b_x[ix, iy] = b[1, iy]
        elseif ix == nx1
            b_x[ix, iy] = b[nx1-1, iy]
        else
            b_x[ix, iy] = (b[ix, iy] + b[ix-1, iy]) / 2
        end
    end
    return
end
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
compute_b_y!(s::State) = (@parallel compute_b_y_kernel!(s.b_y, s.b); s)

# compute_b! only touches each cell using its own local values (mdot, beta,
# abs_ub, A_visc, N, mask) -- no cross-cell stencil, unlike the sparse
# assembly in linear_system.jl -- so unlike that file, it's a genuine xPU
# kernel, not a CPU-only concession.

@parallel_indices (ix, iy) function compute_b_implicit_kernel!(b, mask, mdot, beta, abs_ub, A_visc, N, rho_i, n_minus_1, dt, b_min)
    if ix <= size(b, 1) && iy <= size(b, 2) && mask[ix, iy] == GROUNDED
        b[ix, iy] = max(b_min,
            (b[ix, iy] + dt * (mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy])) /
            (1 + dt * A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy]))
    end
    return
end

@parallel_indices (ix, iy) function compute_b_explicit_kernel!(b, mask, mdot, beta, abs_ub, A_visc, N, rho_i, n_minus_1, dt, b_min)
    if ix <= size(b, 1) && iy <= size(b, 2) && mask[ix, iy] == GROUNDED
        b[ix, iy] = max(b_min,
            b[ix, iy] + dt * (mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy] -
                A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * b[ix, iy]))
    end
    return
end

"""
    compute_b!(sim)

Only evolves the gap height `b` where hydrology is actually being solved
(GROUNDED). Cells with a Dirichlet-prescribed pw (LAND/OCEAN) or a frozen h
(OTHER_BASIN) don't have a meaningfully-evolving `b` in this model, so their
`b` is simply left untouched at whatever it was initialized to.

Dispatches on `sim.gs` (`ImplicitGapScheme()`/`ExplicitGapScheme()`) instead
of branching on a Symbol, so there's no `error("Unknown scheme ...")`
fallback to reach -- the type system already guarantees `sim.gs` is one of
the two.
"""
compute_b!(sim::Simulation) = compute_b!(sim, sim.gs)

function compute_b!(sim::Simulation, ::ImplicitGapScheme)
    s, p = sim.state, sim.p
    @parallel compute_b_implicit_kernel!(s.b, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt, p.b_min)
    return sim
end

function compute_b!(sim::Simulation, ::ExplicitGapScheme)
    s, p = sim.state, sim.p
    @parallel compute_b_explicit_kernel!(s.b, s.mask, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, p.rho_i, p.n_minus_1_exp, sim.dt, p.b_min)
    return sim
end