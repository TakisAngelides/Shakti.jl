# Cubic (Poiseuille) flux law through a sheet of gap height b, with an
# omega*Re term giving a Darcy-Weisbach-style correction towards turbulent
# (as opposed to laminar) flow resistance as Re grows. compute_K! below uses
# the identical law (it's q_x/q_y's coefficient of -dhdx/-dhdy).
@parallel_indices (ix, iy) function compute_q_x_kernel!(q_x, b_x, dhdx, Re_x, ggrav, nu, omega)
    if ix <= size(q_x, 1) && iy <= size(q_x, 2)
        q_x[ix, iy] = -(b_x[ix, iy]^3 * ggrav * dhdx[ix, iy]) / (12 * nu * (1 + omega * Re_x[ix, iy]))
    end
    return
end
compute_q_x!(s::State, p::ModelParameters) = (@parallel compute_q_x_kernel!(s.q_x, s.b_x, s.dhdx, s.Re_x, p.g, p.nu, p.omega); s)

@parallel_indices (ix, iy) function compute_q_y_kernel!(q_y, b_y, dhdy, Re_y, ggrav, nu, omega)
    if ix <= size(q_y, 1) && iy <= size(q_y, 2)
        q_y[ix, iy] = -(b_y[ix, iy]^3 * ggrav * dhdy[ix, iy]) / (12 * nu * (1 + omega * Re_y[ix, iy]))
    end
    return
end
compute_q_y!(s::State, p::ModelParameters) = (@parallel compute_q_y_kernel!(s.q_y, s.b_y, s.dhdy, s.Re_y, p.g, p.nu, p.omega); s)

# Fused hot-path version: one launch instead of two (see field_gradients.jl's
# compute_dhdxy! for why passing both differently-shaped face arrays as
# arguments makes ParallelStencil infer the right union launch range).
@parallel_indices (ix, iy) function compute_q_xy_kernel!(q_x, q_y, b_x, b_y, dhdx, dhdy, Re_x, Re_y, ggrav, nu, omega)
    if ix <= size(q_x, 1) && iy <= size(q_x, 2)
        q_x[ix, iy] = -(b_x[ix, iy]^3 * ggrav * dhdx[ix, iy]) / (12 * nu * (1 + omega * Re_x[ix, iy]))
    end
    if ix <= size(q_y, 1) && iy <= size(q_y, 2)
        q_y[ix, iy] = -(b_y[ix, iy]^3 * ggrav * dhdy[ix, iy]) / (12 * nu * (1 + omega * Re_y[ix, iy]))
    end
    return
end
compute_q_xy!(s::State, p::ModelParameters) = (@parallel compute_q_xy_kernel!(s.q_x, s.q_y, s.b_x, s.b_y, s.dhdx, s.dhdy, s.Re_x, s.Re_y, p.g, p.nu, p.omega); s)

@parallel_indices (ix, iy) function compute_Re_x_kernel!(Re_x, q_x, nu)
    if ix <= size(Re_x, 1) && iy <= size(Re_x, 2)
        Re_x[ix, iy] = abs(q_x[ix, iy]) / nu
    end
    return
end
compute_Re_x!(s::State, p::ModelParameters) = (@parallel compute_Re_x_kernel!(s.Re_x, s.q_x, p.nu); s)

@parallel_indices (ix, iy) function compute_Re_y_kernel!(Re_y, q_y, nu)
    if ix <= size(Re_y, 1) && iy <= size(Re_y, 2)
        Re_y[ix, iy] = abs(q_y[ix, iy]) / nu
    end
    return
end
compute_Re_y!(s::State, p::ModelParameters) = (@parallel compute_Re_y_kernel!(s.Re_y, s.q_y, p.nu); s)

@parallel_indices (ix, iy) function compute_Re_xy_kernel!(Re_x, Re_y, q_x, q_y, nu)
    if ix <= size(Re_x, 1) && iy <= size(Re_x, 2)
        Re_x[ix, iy] = abs(q_x[ix, iy]) / nu
    end
    if ix <= size(Re_y, 1) && iy <= size(Re_y, 2)
        Re_y[ix, iy] = abs(q_y[ix, iy]) / nu
    end
    return
end
compute_Re_xy!(s::State, p::ModelParameters) = (@parallel compute_Re_xy_kernel!(s.Re_x, s.Re_y, s.q_x, s.q_y, p.nu); s)

@parallel_indices (ix, iy) function compute_Re_kernel!(Re, Re_x, Re_y)
    if ix <= size(Re, 1) && iy <= size(Re, 2)
        Re[ix, iy] = (Re_x[ix, iy] + Re_x[ix+1, iy] + Re_y[ix, iy] + Re_y[ix, iy+1]) / 4
    end
    return
end
compute_Re!(s::State) = (@parallel compute_Re_kernel!(s.Re, s.Re_x, s.Re_y); s)

# =============================================================================
# Exact (lag-free) q/Re solve
# =============================================================================
# compute_q_x!/compute_Re_x! above are a LAGGED pair: q is computed from
# whatever Re the *previous* Picard iteration left behind, then Re is
# immediately updated to match the new q -- so q and Re are always exactly
# consistent with EACH OTHER, but q is one iteration stale relative to the
# current dhdx/b. That lag is a real defect, not just theoretically fragile:
# for large gradients it can lock into a period-2 oscillation around the true
# fixed point instead of converging to it at all (confirmed empirically,
# see git history on the exact-q-re-solve branch). Still used for the
# one-shot initial-condition seed (initial_conditions.jl) and in
# test/runtests.jl's setup, where a lagged initial guess is harmless -- it's
# immediately superseded by the first real Picard iteration either way.
#
# q and Re are solved for here with NO lag, given only the current b/dhdx:
# substituting Re = |q|/nu into Eq. 5 (SHAKTI paper, Sommers et al. 2018)
# gives a quadratic. Solved for Re (equivalent up to a change of variables to
# solving for |q| directly -- verified to agree to floating-point precision,
# see git history -- Re is kept since it's the more directly interpretable
# unknown): let D = b^3*g*|dhdx| / (12*nu^2). Then
#   Re = D / (1 + omega*Re)  =>  omega*Re^2 + Re - D = 0
#   Re = (-1 + sqrt(1 + 4*omega*D)) / (2*omega)
# rearranged (multiply by the conjugate) to avoid catastrophic cancellation
# as omega -> 0 (exactly our regime: omega = 1e-4 to 1e-3):
#   Re = 2*D / (1 + sqrt(1 + 4*omega*D))
@parallel_indices (ix, iy) function compute_q_and_Re_x_kernel!(q_x, Re_x, b_x, dhdx, ggrav, nu, omega)
    if ix <= size(q_x, 1) && iy <= size(q_x, 2)
        G = abs(dhdx[ix, iy])
        D = (b_x[ix, iy]^3 * ggrav * G) / (12 * nu^2)
        Re = 2 * D / (1 + sqrt(1 + 4 * omega * D))
        Re_x[ix, iy] = Re
        q_x[ix, iy] = -sign(dhdx[ix, iy]) * nu * Re
    end
    return
end
compute_q_and_Re_x!(s::State, p::ModelParameters) = (@parallel compute_q_and_Re_x_kernel!(s.q_x, s.Re_x, s.b_x, s.dhdx, p.g, p.nu, p.omega); s)

@parallel_indices (ix, iy) function compute_q_and_Re_y_kernel!(q_y, Re_y, b_y, dhdy, ggrav, nu, omega)
    if ix <= size(q_y, 1) && iy <= size(q_y, 2)
        G = abs(dhdy[ix, iy])
        D = (b_y[ix, iy]^3 * ggrav * G) / (12 * nu^2)
        Re = 2 * D / (1 + sqrt(1 + 4 * omega * D))
        Re_y[ix, iy] = Re
        q_y[ix, iy] = -sign(dhdy[ix, iy]) * nu * Re
    end
    return
end
compute_q_and_Re_y!(s::State, p::ModelParameters) = (@parallel compute_q_and_Re_y_kernel!(s.q_y, s.Re_y, s.b_y, s.dhdy, p.g, p.nu, p.omega); s)

@parallel_indices (ix, iy) function compute_q_and_Re_xy_kernel!(q_x, q_y, Re_x, Re_y, b_x, b_y, dhdx, dhdy, ggrav, nu, omega)
    if ix <= size(q_x, 1) && iy <= size(q_x, 2)
        Gx = abs(dhdx[ix, iy])
        Dx = (b_x[ix, iy]^3 * ggrav * Gx) / (12 * nu^2)
        Rex = 2 * Dx / (1 + sqrt(1 + 4 * omega * Dx))
        Re_x[ix, iy] = Rex
        q_x[ix, iy] = -sign(dhdx[ix, iy]) * nu * Rex
    end
    if ix <= size(q_y, 1) && iy <= size(q_y, 2)
        Gy = abs(dhdy[ix, iy])
        Dy = (b_y[ix, iy]^3 * ggrav * Gy) / (12 * nu^2)
        Rey = 2 * Dy / (1 + sqrt(1 + 4 * omega * Dy))
        Re_y[ix, iy] = Rey
        q_y[ix, iy] = -sign(dhdy[ix, iy]) * nu * Rey
    end
    return
end
compute_q_and_Re_xy!(s::State, p::ModelParameters) = (@parallel compute_q_and_Re_xy_kernel!(s.q_x, s.q_y, s.Re_x, s.Re_y, s.b_x, s.b_y, s.dhdx, s.dhdy, p.g, p.nu, p.omega); s)

# Same cubic/turbulence-corrected law as compute_q_x!/compute_q_y! above,
# but expressed as a hydraulic conductivity (i.e. without the -dhdx/-dhdy
# factor) for use as the linear system's (off-diagonal) coefficients.
@parallel_indices (ix, iy) function compute_K_kernel!(K, b, Re, ggrav, nu, omega)
    if ix <= size(K, 1) && iy <= size(K, 2)
        K[ix, iy] = (b[ix, iy]^3 * ggrav) / (12 * nu * (1 + omega * Re[ix, iy]))
    end
    return
end
compute_K!(s::State, p::ModelParameters) = (@parallel compute_K_kernel!(s.K, s.b, s.Re, p.g, p.nu, p.omega); s)
