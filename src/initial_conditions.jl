function set_initial_conditions!(s::State, g::Grid, p::ModelParameters, mi::AbstractMeltInput, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)

    # State's fields all share one array/element type, so any field's eltype
    # gives the right F -- used below wherever a bare numeric literal would
    # otherwise default to Float64 and get fused into a broadcast over
    # (possibly GPU-resident) State arrays. Under Metal (no hardware double
    # precision) that's not just wasted precision, it risks the broadcast
    # kernel failing to compile outright -- same issue as eps() in
    # elliptic_solver.jl's convergence check.
    F = eltype(s.h)

    @. s.mask = mask
    compute_face_masks!(s, g)

    @. s.A_visc = A_visc
    @. s.lambda = F(1.5) * s.A_visc

    @. s.zb = zb
    @. s.zs = zs

    @. s.b = b
    @inbounds for j in 1:g.ny, i in 1:g.nx
        if s.mask[i, j] != GROUNDED
            s.b[i, j] = F(0.0)
        end
    end

    compute_H!(s)
    compute_beta!(s, p)

    @. s.G = G

    @. s.ub_x = ub_x
    @. s.ub_y = ub_y
    apply_mask_to_sliding!(s, g)
    compute_abs_ub!(s)

    initialize_ieb!(mi, s, ieb)

    compute_po!(s, p)
    @. s.pw = s.po / 2
    # Overwrite pw with the prescribed Dirichlet value wherever the mask calls
    # for one, so the initial condition is consistent with the BC from t=0.
    # Harmless numerically (the Poisson solve overwrites h/pw on those rows
    # immediately) but keeps pw/N consistent from the start.
    @inbounds for j in 1:g.ny, i in 1:g.nx
        if s.mask[i, j] == OCEAN
            # See linear_system.jl's OCEAN branch for the sign convention.
            s.pw[i, j] = p.p_atm - p.rho_w * p.g * min(s.zb[i, j], zero(F))
        elseif s.mask[i, j] == LAND || s.mask[i, j] == OTHER_BASIN
            s.pw[i, j] = p.p_atm
        end
    end
    compute_dpwdx!(s, g) # feeds compute_sensible!'s sensible-heat term (via compute_mdot! below)
    compute_dpwdy!(s, g)
    compute_N!(s)

    compute_h!(s, p)

    compute_dhdx!(s, g)
    compute_dhdy!(s, g)

    @. s.Re_x = F(1000.0)
    @. s.Re_y = F(1000.0)
    compute_Re!(s)

    compute_b_x!(s)
    compute_b_y!(s)

    compute_q_x!(s, p)
    compute_q_y!(s, p)

    compute_taub_x!(s, p)
    compute_taub_y!(s, p)

    # Sensible-heat scheme setup: same "off automatically if either factor in
    # its ct*cw prefactor is zero" rule as Simulation's own constructor.
    shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
    compute_mdot!(s, p, shs)
    compute_K!(s, p)


end