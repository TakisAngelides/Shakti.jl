@parallel_indices (ix, iy) function compute_pw_kernel!(pw, h, zb, rho_w, ggrav)
    if ix <= size(pw, 1) && iy <= size(pw, 2)
        pw[ix, iy] = rho_w * ggrav * (h[ix, iy] - zb[ix, iy])
    end
    return
end
compute_pw!(s::State, p::ModelParameters) = (@parallel compute_pw_kernel!(s.pw, s.h, s.zb, p.rho_w, p.g); s)

@parallel_indices (ix, iy) function compute_N_kernel!(N, po, pw)
    if ix <= size(N, 1) && iy <= size(N, 2)
        N[ix, iy] = po[ix, iy] - pw[ix, iy]
    end
    return
end
compute_N!(s::State) = (@parallel compute_N_kernel!(s.N, s.po, s.pw); s)