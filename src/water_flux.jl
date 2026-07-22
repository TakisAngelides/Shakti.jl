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
