"""
$(TYPEDSIGNATURES)

Populates `s` (built with [`State(::Grid)`](@ref), fields all zero) with a real starting state:
the raw input fields (`mask`, `A_visc`, `zb`, `zs`, `b`, `G`, `ub_x`, `ub_y`, `ieb`, `taub_x`,
`taub_y`) are copied in (converting to `s`'s backend/element type via `Data.Array`), then every
derived field (`H`, `beta`, `abs_ub`, `po`, `pw`, `N`, `h`, gradients, `Re`, `b_x`/`b_y`,
`q_x`/`q_y`, `taub_x`/`taub_y`, `mdot`, `K`) is computed from those in the same order the Picard
loop itself would produce them, so `s` is immediately a valid state to time-step or Picard-solve
from. `pw` is initialized to half of ice overburden pressure (`po/2`) as a generic starting
guess -- not a converged solution, see the elliptic solver for that -- except where the mask
calls for a prescribed Dirichlet value.
"""
function set_initial_conditions!(s::State, g::Grid, p::ModelParameters, sl::AbstractSlidingLaw, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)

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

    @. s.mask = mask # 0.0: grounded ice, 1.0: ocean, 2.0: land, 3.0: other basin - see mask.jl
    compute_face_masks!(s) # goes into mask.jl to calculate valid_x and valid_y for the faces that touch OTHER_BASIN cells to have their field value be zeroed there when we compute e.g. staggered gradients on faces

    @. s.A_visc = A_visc # Glen's flow law viscosity parameter field
    lambda_coeff = F(1.5) # precomputed outside the broadcast: `@.` rewrites every call it sees, including F(1.5) itself, into F.(1.5) -- fusing a Float64-literal broadcast into the kernel and risking the same GPU compile failure this was meant to avoid
    @. s.lambda = lambda_coeff * s.A_visc

    @. s.zb = zb # z elevation of bedrock
    @. s.zs = zs # z elevation of ice surface

    @. s.b = b
    # Vectorized rather than a scalar for-loop: GPUArrays.jl disallows
    # element-by-element getindex!/setindex! on GPU-resident arrays
    # (Metal.MtlArray included) by default, so an @inbounds for/if pattern
    # over s.mask/s.b never runs under the Metal backend.
    zero_b = F(0.0)
    @. s.b = ifelse(s.mask != GROUNDED, zero_b, s.b) # if the cell has no grounded ice the water thickness is initialized to zero

    compute_H!(s) # update the ice thickness

    # Opening-by-sliding scheme setup: same "off automatically when p.br == 0" rule as Simulation's own constructor.
    oss = iszero(p.br) ? NoOpenBySliding() : WithOpenBySliding()
    compute_beta!(s, p, oss) # update beta which is the parameter field used in the opening-by-sliding term

    @. s.G = G # geothermal heat flux

    # Basal velocity
    @. s.ub_x = ub_x
    @. s.ub_y = ub_y
    apply_mask_to_sliding!(s) # valid_x, valid_y are multiplied on the ub_x and ub_y face fields respectively to zero out any face with a non-zero velocity that touches a cell which is OTHER_BASIN, this should be called every time the ub fields are updated e.g. from an ice flow model
    compute_abs_ub!(s) # this is the magnitude of the basal velocity vector

    s.ieb .= ieb # englacial to subglacial water input i₍e → b₎, seeded as-is; per-timestep evolution (if any) is handled by update_ieb!

    initialize_taub!(sl, s, taub_x, taub_y) # basal shear stress, the `sl` determines the sliding law

    compute_po!(s, p) # ice overburden pressure
    @. s.pw = s.po / 2 # water pressure
    # Overwrite pw with the prescribed Dirichlet value wherever the mask calls
    # for one, so the initial condition is consistent with the BC from t=0.
    # Harmless numerically (the Poisson solve overwrites h/pw on those rows
    # immediately) but keeps pw/N consistent from the start.
    # Vectorized rather than a scalar for-loop so that it is GPU compatible.
    zero_zb = zero(F)
    @. s.pw = ifelse(s.mask == OCEAN, # if the cell is OCEAN then set the water pressure to be the hydrostatic pressure of ocean water that is present above the zb at that point
                      p.p_atm - p.rho_sw * p.g * min(s.zb, zero_zb), # see linear_system.jl's OCEAN branch for the sign convention
               ifelse((s.mask == LAND) | (s.mask == OTHER_BASIN),
                      p.p_atm, s.pw))
    compute_dpwdx!(s, g) # water pressure gradient in x and y, feeds compute_sensible!'s sensible-heat term (via compute_mdot! below)
    compute_dpwdy!(s, g)
    compute_N!(s) # Effective pressure

    compute_h!(s, p) # Hydraulic head h = pw/(rho_w * g) + zb

    # Gradients of hydraulic head
    compute_dhdx!(s, g)
    compute_dhdy!(s, g)

    # Set the face fields of the water depth according to the center fields of the water depth via simple arithmetic mean interpolation
    compute_b_x!(s)
    compute_b_y!(s)

    # Water flux and Reynolds number: solved exactly (no arbitrary initial
    # guess, no lag) from the quadratic in Re that Eq. 5, 7 (Sommers et al. 2018)
    # reduces to once Re = |q|/nu is substituted in -- see
    # compute_q_and_Re_x!'s docstring in water_flux.jl. Same pair of calls
    # Picard_iteration! uses each iteration, so q_x/q_y/Re start already
    # consistent with the initial b_x/b_y/dhdx/dhdy, instead of the first
    # real Picard iteration having to correct an arbitrary Re guess.
    compute_q_and_Re_xy!(s, p)
    compute_Re!(s) # cell-centered average of Re_x/Re_y, feeds compute_K! below

    # Basal shear stress
    compute_taub_x!(s, p, sl)
    compute_taub_y!(s, p, sl)

    # Sensible-heat scheme setup: same "off automatically if either factor in
    # its ct*cw prefactor is zero" rule as Simulation's own constructor.
    shs = (iszero(p.ct) || iszero(p.cw)) ? NoSensibleHeat() : WithSensibleHeat()
    compute_mdot!(s, p, shs) # melt rate
    compute_K!(s, p) # transmissivity

end