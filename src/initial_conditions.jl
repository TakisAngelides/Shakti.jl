function set_initial_conditions!(s::State, g::Grid, p::ModelParameters, mi::AbstractMeltInput, sl::AbstractSlidingLaw, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

    # State's fields all share one array/element type, so any field's eltype
    # gives the right F -- used below wherever a bare numeric literal would
    # otherwise default to Float64 and get fused into a broadcast over
    # (possibly GPU-resident) State arrays. Under Metal (no hardware double
    # precision) that's not just wasted precision, it risks the broadcast
    # kernel failing to compile outright -- same issue as eps() in
    # elliptic_solver.jl's convergence check.
    F = eltype(s.h)

    # Inputs arrive as plain, host-resident arrays (however the caller built
    # them) regardless of backend. Under Metal, broadcasting a plain CPU
    # array directly into a GPU-resident State field doesn't work at all --
    # GPU broadcast fusion requires every operand to already live on the
    # device. Data.Array (Threads -> Array, Metal -> MtlArray) does both the
    # device placement and the eltype conversion in one step, matching s.h's
    # own storage.
    mask   = Data.Array(mask)
    A_visc = Data.Array(A_visc)
    zb     = Data.Array(zb)
    zs     = Data.Array(zs)
    b      = Data.Array(b)
    G      = Data.Array(G)
    ub_x   = Data.Array(ub_x)
    ub_y   = Data.Array(ub_y)
    ieb    = Data.Array(ieb)
    taub_x = Data.Array(taub_x)
    taub_y = Data.Array(taub_y)

    @. s.mask = mask
    compute_face_masks!(s)

    @. s.A_visc = A_visc
    lambda_coeff = F(1.5) # precomputed outside the broadcast: `@.` rewrites every call it sees, including F(1.5) itself, into F.(1.5) -- fusing a Float64-literal broadcast into the kernel and risking the same GPU compile failure this was meant to avoid
    @. s.lambda = lambda_coeff * s.A_visc

    @. s.zb = zb
    @. s.zs = zs

    @. s.b = b
    # Vectorized rather than a scalar for-loop: GPUArrays.jl disallows
    # element-by-element getindex!/setindex! on GPU-resident arrays
    # (Metal.MtlArray included) by default, so an @inbounds for/if pattern
    # over s.mask/s.b never runs under the Metal backend.
    zero_b = F(0.0)
    @. s.b = ifelse(s.mask != GROUNDED, zero_b, s.b)

    compute_H!(s)
    compute_beta!(s, p)

    @. s.G = G

    @. s.ub_x = ub_x
    @. s.ub_y = ub_y
    apply_mask_to_sliding!(s)
    compute_abs_ub!(s)

    initialize_ieb!(mi, s, ieb)
    initialize_taub!(sl, s, taub_x, taub_y)

    compute_po!(s, p)
    @. s.pw = s.po / 2
    # Overwrite pw with the prescribed Dirichlet value wherever the mask calls
    # for one, so the initial condition is consistent with the BC from t=0.
    # Harmless numerically (the Poisson solve overwrites h/pw on those rows
    # immediately) but keeps pw/N consistent from the start.
    # Vectorized rather than a scalar for-loop, same reason as s.b above.
    zero_zb = zero(F)
    @. s.pw = ifelse(s.mask == OCEAN,
                      p.p_atm - p.rho_w * p.g * min(s.zb, zero_zb), # see linear_system.jl's OCEAN branch for the sign convention
               ifelse((s.mask == LAND) | (s.mask == OTHER_BASIN),
                      p.p_atm, s.pw))
    compute_dpwdx!(s, g) # feeds compute_sensible!'s sensible-heat term (via compute_mdot! below)
    compute_dpwdy!(s, g)
    compute_N!(s)

    compute_h!(s, p)

    compute_dhdx!(s, g)
    compute_dhdy!(s, g)

    re_init = F(1000.0)
    @. s.Re_x = re_init
    @. s.Re_y = re_init
    compute_Re!(s)

    compute_b_x!(s)
    compute_b_y!(s)

    compute_q_x!(s, p)
    compute_q_y!(s, p)

    compute_taub_x!(s, p, sl)
    compute_taub_y!(s, p, sl)

    # Sensible-heat scheme setup: same "off automatically if either factor in
    # its ct*cw prefactor is zero" rule as Simulation's own constructor.
    shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
    compute_mdot!(s, p, shs)
    compute_K!(s, p)

end