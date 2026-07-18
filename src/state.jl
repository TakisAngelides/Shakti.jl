struct State{A <: AbstractArray, V <: AbstractVector}

    # Center fields
    h::A          # hydraulic head
    h_prev::A     # previous hydraulic head
    h_vec::V      # vectorized hydraulic head
    pw::A         # water pressure
    po::A         # ice overburden pressure
    b::A          # water depth
    beta::A       # parameter for opening by sliding over bedrock bumps
    abs_ub::A     # absolute value of the sliding velocity
    mdot::A       # melt rate
    shear::A      # frictional (sliding) heating contribution to mdot, preallocated for compute_mdot!
    potential::A  # potential-energy-dissipation contribution to mdot, preallocated for compute_mdot!
    sensible::A   # sensible-heat-exchange contribution to mdot, preallocated for compute_mdot!
    Re::A         # Reynolds number
    K::A          # hydraulic conductivity
    G::A          # geothermal heat flux
    zb::A         # bedrock elevation
    zs::A         # ice surface elevation
    H::A          # ice thickness
    ieb::A        # input from moulins
    lambda::A     # ratio of controlling bedrock bump wavelength to maximum slope
    A_visc::A     # Glen's flow law rate factor
    N::A          # effective pressure
    mask::A       # 0 grounded, 1 ocean, 2 land, 3 other grounded-ice basin (static per run)
    delta_h::A    # change in hydraulic head for convergence check

    # XFace fields
    dhdx::A       # gradient of hydraulic head in x direction
    q_x::A        # water flux in x direction
    Re_x::A       # Reynolds number in x direction
    b_x::A        # water depth in x direction
    ub_x::A       # sliding velocity in x direction
    taub_x::A     # basal shear stress in x direction
    dpwdx::A      # gradient of water pressure in x direction
    valid_x::A    # 1.0 where the x-face does NOT touch an OTHER_BASIN cell, else 0.0

    # YFace fields
    dhdy::A       # gradient of hydraulic head in y direction
    q_y::A        # water flux in y direction
    Re_y::A       # Reynolds number in y direction
    b_y::A        # water depth in y direction
    ub_y::A       # sliding velocity in y direction
    taub_y::A     # basal shear stress in y direction
    dpwdy::A      # gradient of water pressure in y direction
    valid_y::A    # 1.0 where the y-face does NOT touch an OTHER_BASIN cell, else 0.0
end

initialize_center_field(g::Grid) = @zeros(g.nx, g.ny)
initialize_xface_field(g::Grid)  = @zeros(g.nx + 1, g.ny)
initialize_yface_field(g::Grid)  = @zeros(g.nx, g.ny + 1)

function State(g::Grid)

    # Center fields
    h         = initialize_center_field(g)
    h_prev    = initialize_center_field(g)
    h_vec     = @zeros(g.nx * g.ny)
    pw        = initialize_center_field(g)
    po        = initialize_center_field(g)
    b         = initialize_center_field(g)
    beta      = initialize_center_field(g)
    abs_ub    = initialize_center_field(g)
    mdot      = initialize_center_field(g)
    shear     = initialize_center_field(g)
    potential = initialize_center_field(g)
    sensible  = initialize_center_field(g)
    Re        = initialize_center_field(g)
    K         = initialize_center_field(g)
    G         = initialize_center_field(g)
    zb        = initialize_center_field(g)
    zs        = initialize_center_field(g)
    H         = initialize_center_field(g)
    ieb       = initialize_center_field(g)
    lambda    = initialize_center_field(g)
    A_visc    = initialize_center_field(g)
    N         = initialize_center_field(g)
    mask      = @fill(0, g.nx, g.ny) # 0: GROUNDED, 1: OCEAN, 2: LAND, 3: OTHER_BASIN
    delta_h   = initialize_center_field(g)

    # XFace fields
    dhdx    = initialize_xface_field(g)
    q_x     = initialize_xface_field(g)
    Re_x    = initialize_xface_field(g)
    b_x     = initialize_xface_field(g)
    ub_x    = initialize_xface_field(g)
    taub_x  = initialize_xface_field(g)
    dpwdx   = initialize_xface_field(g)
    valid_x = @fill(1, g.nx+1, g.ny) # float 1.0 = valid; recomputed in compute_face_masks!

    # YFace fields
    dhdy    = initialize_yface_field(g)
    q_y     = initialize_yface_field(g)
    Re_y    = initialize_yface_field(g)
    b_y     = initialize_yface_field(g)
    ub_y    = initialize_yface_field(g)
    taub_y  = initialize_yface_field(g)
    dpwdy   = initialize_yface_field(g)
    valid_y = @fill(1, g.nx, g.ny+1) # float 1.0 = valid; recomputed in compute_face_masks!

    return State(
        h, h_prev, h_vec, pw, po, b, beta, abs_ub, mdot, shear, potential, sensible, Re, K, G, zb, zs, H, ieb, lambda, A_visc, N, mask, delta_h,
        dhdx, q_x, Re_x, b_x, ub_x, taub_x, dpwdx, valid_x,
        dhdy, q_y, Re_y, b_y, ub_y, taub_y, dpwdy, valid_y,
    )

end