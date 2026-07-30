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

@parallel_indices (ix, iy) function compute_N_kernel!(N, po, pw, mask)
    if ix <= size(N, 1) && iy <= size(N, 2)
        is_grounded = mask[ix, iy] == GROUNDED # float comparison, not Bool -- see mask.jl's header note
        N[ix, iy] = is_grounded * (po[ix, iy] - pw[ix, iy])
    end
    return
end

"""
$(TYPEDSIGNATURES)

Updates the `s.N` (effective pressure) from the current `s.po`/`s.pw`, restricted to `GROUNDED`
cells: `N = po - pw` where grounded ice is present, `0` everywhere else (`OCEAN`/`LAND`/
`OTHER_BASIN`). Effective pressure is only a physically meaningful quantity under grounded ice;
without this mask, non-`GROUNDED` cells would report whatever `po - pw` happens to work out to
from their Dirichlet/frozen boundary conditions (e.g. `OCEAN`'s hydrostatic `pw` against `po = 0`),
which is a real number but not effective pressure.
"""
compute_N!(s::State) = (@parallel compute_N_kernel!(s.N, s.po, s.pw, s.mask); s)