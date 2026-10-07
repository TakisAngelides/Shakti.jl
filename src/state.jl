"""
$(TYPEDSIGNATURES)

Every field of the subglacial hydrology solve: cell-centered quantities (hydraulic head, water
and overburden pressure, effective pressure, gap height, melt rate and its three heat-source
components, ...) and the x-/y-face quantities needed for the finite-difference flux/gradient stencils
(`Nx+1 x Ny` and `Nx x Ny+1` respectively). Every field shares one array/element type `A`, so
switching backend or floating-point precision (see the `Preferences`-backed `backend`/`floattype`
constants in `Shakti.jl`) only ever touches [`Grid`](@ref)/`State` construction, not the kernels
that operate on them. Build one with `State(grid)`, then populate it via
[`set_initial_conditions!`](@ref).
"""
struct State{A <: AbstractArray}

    # Center fields
    h::A          # hydraulic head
    pw::A         # water pressure
    po::A         # ice overburden pressure
    b::A          # water depth
    b_w::A        # water depth at the current head iterate [m] (b - empty part, floored); what the face conductances are built from (unfilled_cavities.jl). Equals b exactly under FilledCavities
    w_old::A      # water per bed area [m] at the start of the step, b - b_empty (UnfilledCavities, see update_b_empty_budget!)
    stor::A       # water storage coefficient of an unfilled cavity [1/s]: 1/dt on GROUNDED cells under UnfilledCavities, 0 otherwise (0 = legacy, always-filled gap)
    b_empty::A    # empty (water-free) part of the gap per unit bed area [m], >= 0; only used under UnfilledCavities (unfilled_cavities.jl), zero otherwise. Water actually present is b - b_empty
    beta::A       # parameter for opening by sliding over bedrock bumps
    lc::A         # ice-creep length scale (see AbstractCreepLengthScheme, gap_height.jl); equals b under StandardCreep
    abs_ub::A     # absolute value of the sliding velocity
    mdot::A       # melt rate
    mdot_min::A   # floor on mdot this step, -rho_i*C (freeze-on capacity); applied when p.limit_freeze_on
    Q_b::A        # frictional (sliding) heat |u_b . taub| [W/m^2], written by compute_mdot! (0 when that term is off)
    Q_diss::A     # heat dissipated by water flowing down the head gradient rho_w*g*|q . grad(h)| [W/m^2] (0 when off)
    Q_sens::A     # sensible heat ct*cw*rho_w*(q . grad(pw)) [W/m^2] (0 when off)
    Re::A         # Reynolds number
    K::A          # hydraulic conductivity
    G::A          # geothermal heat flux
    q_T::A        # conductive heat flux escaping into cold ice above the bed (e.g. Yelmo's
                  # Q_ice_b), subtracted from mdot's numerator in compute_mdot!. Zero by default
                  # (no effect on mdot), so it is opt-in per run/dataset.
    zb::A         # bedrock elevation
    zs::A         # ice surface elevation
    H::A          # ice thickness
    ieb::A        # englacial-to-bed water input i_e->b [m/s] used by the solve: ieb_own + ieb_external (combine_ieb!)
    ieb_own::A    # Shakti's own part of ieb: seeded by set_initial_conditions! (ConstantMeltInput) or written by update_ieb! (SeasonalMeltInput, GaussianMoulinMeltInput)
    ieb_external::A # part of ieb supplied by a coupled model (e.g. Yelmo's englacial drainage), set with set_ieb_external!; zero when Shakti runs standalone
    lambda::A     # ratio of controlling bedrock bump wavelength to maximum slope
    A_visc::A     # Glen's flow law rate factor
    N::A          # effective pressure
    mask::A       # 0 grounded, 1 ocean, 2 land, 3 other grounded-ice basin (static per run)

    # XFace fields
    dhdx::A       # gradient of hydraulic head in x direction
    q_x::A        # water flux in x direction
    Re_x::A       # Reynolds number in x direction
    b_x::A        # water depth in x direction
    ub_x::A       # sliding velocity in x direction
    taub_x::A     # basal shear stress in x direction
    dpwdx::A      # gradient of water pressure in x direction
    D_x::A        # channel-wall diffusion coefficient in x direction (SUHMO Eq. 10, WithDiffusion only -- see AbstractDiffusionScheme, linear_solver.jl)
    K_x::A        # face transmissivity in x direction: what the linear system assembles and q_x is built from (compute_face_flux!, water_flux.jl)
    valid_x::A    # 1.0 where the x-face does NOT touch an OTHER_BASIN or FROZEN_BED cell, else 0.0

    # YFace fields
    dhdy::A       # gradient of hydraulic head in y direction
    q_y::A        # water flux in y direction
    Re_y::A       # Reynolds number in y direction
    b_y::A        # water depth in y direction
    ub_y::A       # sliding velocity in y direction
    taub_y::A     # basal shear stress in y direction
    dpwdy::A      # gradient of water pressure in y direction
    D_y::A        # channel-wall diffusion coefficient in y direction, the y-face counterpart of D_x
    K_y::A        # face transmissivity in y direction, the y-face counterpart of K_x
    valid_y::A    # 1.0 where the y-face does NOT touch an OTHER_BASIN or FROZEN_BED cell, else 0.0
end

"""
$(TYPEDSIGNATURES)

Allocates a zeroed cell-centered (`nx x ny`) field on `g`'s backend.
"""
initialize_center_field(g::Grid) = @zeros(g.nx, g.ny)

"""
$(TYPEDSIGNATURES)

Allocates a zeroed x-face (`nx+1 x ny`) field on `g`'s backend.
"""
initialize_xface_field(g::Grid)  = @zeros(g.nx + 1, g.ny)

"""
$(TYPEDSIGNATURES)

Allocates a zeroed y-face (`nx x ny+1`) field on `g`'s backend.
"""
initialize_yface_field(g::Grid)  = @zeros(g.nx, g.ny + 1)

"""
$(TYPEDSIGNATURES)

Builds a [`State`](@ref) on `g`, with every field zeroed (`mask` defaults to all-`GROUNDED`) --
see [`set_initial_conditions!`](@ref) to populate it with real data.
"""
function State(g::Grid)

    # Center fields
    h         = initialize_center_field(g)
    pw        = initialize_center_field(g)
    po        = initialize_center_field(g)
    b         = initialize_center_field(g)
    b_empty   = initialize_center_field(g)
    b_w       = initialize_center_field(g)
    stor      = initialize_center_field(g)
    w_old     = initialize_center_field(g)
    beta      = initialize_center_field(g)
    lc        = initialize_center_field(g)
    abs_ub    = initialize_center_field(g)
    mdot      = initialize_center_field(g)
    mdot_min  = @fill(-Inf, g.nx, g.ny) # no floor until prepare_head_solve! sets one
    Q_b       = initialize_center_field(g)
    Q_diss    = initialize_center_field(g)
    Q_sens    = initialize_center_field(g)
    Re        = initialize_center_field(g)
    K         = initialize_center_field(g)
    G         = initialize_center_field(g)
    q_T       = initialize_center_field(g)
    zb        = initialize_center_field(g)
    zs        = initialize_center_field(g)
    H         = initialize_center_field(g)
    ieb       = initialize_center_field(g)
    ieb_own   = initialize_center_field(g)
    ieb_external = initialize_center_field(g)
    lambda    = initialize_center_field(g)
    A_visc    = initialize_center_field(g)
    N         = initialize_center_field(g)
    mask      = @fill(0.0, g.nx, g.ny) # 0.0: GROUNDED, 1.0: OCEAN, 2.0: LAND, 3.0: OTHER_BASIN, 4.0: FROZEN_BED

    # XFace fields
    dhdx    = initialize_xface_field(g)
    q_x     = initialize_xface_field(g)
    Re_x    = initialize_xface_field(g)
    b_x     = initialize_xface_field(g)
    ub_x    = initialize_xface_field(g)
    taub_x  = initialize_xface_field(g)
    dpwdx   = initialize_xface_field(g)
    D_x     = initialize_xface_field(g)
    K_x     = initialize_xface_field(g)
    valid_x = @fill(1.0, g.nx+1, g.ny) # float 1.0 = valid; recomputed in compute_face_masks!

    # YFace fields
    dhdy    = initialize_yface_field(g)
    q_y     = initialize_yface_field(g)
    Re_y    = initialize_yface_field(g)
    b_y     = initialize_yface_field(g)
    ub_y    = initialize_yface_field(g)
    taub_y  = initialize_yface_field(g)
    dpwdy   = initialize_yface_field(g)
    D_y     = initialize_yface_field(g)
    K_y     = initialize_yface_field(g)
    valid_y = @fill(1.0, g.nx, g.ny+1) # float 1.0 = valid; recomputed in compute_face_masks!

    return State(
        h, pw, po, b, b_w, w_old, stor, b_empty, beta, lc, abs_ub, mdot, mdot_min, Q_b, Q_diss, Q_sens, Re, K, G, q_T, zb, zs, H, ieb, ieb_own, ieb_external, lambda, A_visc, N, mask,
        dhdx, q_x, Re_x, b_x, ub_x, taub_x, dpwdx, D_x, K_x, valid_x,
        dhdy, q_y, Re_y, b_y, ub_y, taub_y, dpwdy, D_y, K_y, valid_y,
    )

end