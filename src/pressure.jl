@parallel_indices (ix, iy) function compute_pw_kernel!(pw, h, zb, rho_w, ggrav)
    if ix <= size(pw, 1) && iy <= size(pw, 2)
        pw[ix, iy] = rho_w * ggrav * (h[ix, iy] - zb[ix, iy])
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates the `s.pw` (water pressure) from the current `s.h` (hydraulic head): `pw = rho_w*g*(h - zb)`.
"""
compute_pw!(s::State, p::ModelParameters) = (@parallel compute_pw_kernel!(s.pw, s.h, s.zb, p.rho_w, p.g); s)

@parallel_indices (ix, iy) function compute_N_kernel!(N, po, pw, mask, N_min, N_max)
    if ix <= size(N, 1) && iy <= size(N, 2)
        is_grounded = mask[ix, iy] == GROUNDED # Float64 == Float64 (not Int), matching mask.jl's own rationale for keeping mask float-valued
        N[ix, iy] = is_grounded * clamp(po[ix, iy] - pw[ix, iy], N_min, N_max)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates the `s.N` (effective pressure) from the current `s.po`/`s.pw`, restricted to `GROUNDED`
cells: `N = clamp(po - pw, p.N_min, p.N_max)` where grounded ice is present, `0` everywhere else
(`OCEAN`/`LAND`/`OTHER_BASIN`). Effective pressure is only a physically meaningful quantity under
grounded ice; without this mask, non-`GROUNDED` cells would report whatever `po - pw` happens to
work out to from their Dirichlet/frozen boundary conditions (e.g. `OCEAN`'s hydrostatic `pw`
against `po = 0`), which is a real number but not effective pressure. `N_min`/`N_max` default to
`-Inf`/`Inf` (no-op clamp, see [`ModelParameters`](@ref)'s docstring). `cnc` (default
[`NoCellNClamping`](@ref)) additionally applies a per-cell override on top, see
[`apply_cell_N__clamping!`](@ref).
"""
function compute_N!(s::State, p::ModelParameters, cnc::AbstractCellNClamping = NoCellNClamping())
    @parallel compute_N_kernel!(s.N, s.po, s.pw, s.mask, p.N_min, p.N_max)
    apply_cell_N__clamping!(s, cnc)
    return s
end