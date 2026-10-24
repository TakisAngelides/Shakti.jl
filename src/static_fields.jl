# =============================================================================
# Static / initial geometry fields
# =============================================================================
# Computed once per time step rather than every Picard iteration: set_initial_conditions! seeds them, and every step
# refreshes H/po from the current b, zb, zs at its start (prepare_head_solve!, run.jl), so a coupled ice model's new
# geometry is picked up; they stay fixed during the step's head solve.

# The ice base sits b above the bed, and under IceUplift a further W - b where the water lifts it (the uplift layer,
# water_above_gap); the ice above it is floored at zero thickness, so a gap or layer thicker than zs - zb (a thin
# margin) gives po = 0, not a negative overburden that would cap pw below zero and read the closure at N < 0.
@parallel_indices (ix, iy) function compute_H_kernel!(H, zs, zb, b, W, lifted)
    if ix <= size(H, 1) && iy <= size(H, 2)
        bu = lifted ? max(zero(eltype(H)), W[ix, iy] - b[ix, iy]) : zero(eltype(H))
        H[ix, iy] = max(zero(eltype(H)), zs[ix, iy] - (zb[ix, iy] + b[ix, iy] + bu))
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.H` (ice thickness) from the current `s.zs`/`s.zb`/`s.b`: `H = max(0, zs - (zb + b + b_uplift))`, the ice
above the gap and, under [`IceUplift`](@ref) (`up`), above the uplift layer `b_uplift = max(0, W - b)`
([`water_above_gap`](@ref)); `b_uplift = 0` under [`NoUplift`](@ref) or with no `up` given. Floored at zero, so a gap
thicker than the ice (a thin margin) gives no negative overburden.
"""
compute_H!(s::State, up::AbstractIceUplift = NoUplift()) = (@parallel compute_H_kernel!(s.H, s.zs, s.zb, s.b, s.W, up isa IceUplift); s)

@parallel_indices (ix, iy) function compute_po_kernel!(po, H, rho_i, ggrav)
    if ix <= size(po, 1) && iy <= size(po, 2)
        po[ix, iy] = rho_i * ggrav * H[ix, iy]
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.po` (ice overburden pressure) from the current `s.H`: `po = rho_i*g*H`.
"""
compute_po!(s::State, p::ModelParameters) = (@parallel compute_po_kernel!(s.po, s.H, p.rho_i, p.g); s)

# Inverse of compute_pw! below: seeds h from a prescribed initial pw. Once the
# time loop is running, h is instead the Picard solver's primary unknown, and
# pw is derived FROM h every iteration (compute_pw!), not the other way round.
@parallel_indices (ix, iy) function compute_h_kernel!(h, pw, zb, rho_w, ggrav)
    if ix <= size(h, 1) && iy <= size(h, 2)
        h[ix, iy] = pw[ix, iy] / (rho_w * ggrav) + zb[ix, iy]
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.h` (hydraulic head) from the current `s.pw`: `h = pw/(rho_w*g) + zb`.

# Notes

Only meant for seeding `s.h` from a prescribed initial `s.pw` during initial-condition setup.
Once the time loop is running, `h` is instead the Picard solver's primary unknown, and `pw` is
derived FROM `h` every iteration ([`compute_pw!`](@ref)), not the other way round.
"""
compute_h!(s::State, p::ModelParameters) = (@parallel compute_h_kernel!(s.h, s.pw, s.zb, p.rho_w, p.g); s)

@parallel_indices (ix, iy) function compute_abs_ub_kernel!(abs_ub, ub_x, ub_y)
    if ix <= size(abs_ub, 1) && iy <= size(abs_ub, 2)
        ubx_c = (ub_x[ix, iy] + ub_x[ix+1, iy]) / 2
        uby_c = (ub_y[ix, iy] + ub_y[ix, iy+1]) / 2
        abs_ub[ix, iy] = sqrt(ubx_c^2 + uby_c^2)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates `s.abs_ub` (cell-centered sliding speed magnitude `‖v_b‖`) by averaging `s.ub_x`/`s.ub_y`
onto cell centers and taking the norm.
"""
compute_abs_ub!(s::State) = (@parallel compute_abs_ub_kernel!(s.abs_ub, s.ub_x, s.ub_y); s)