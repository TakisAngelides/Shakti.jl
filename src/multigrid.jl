# =============================================================================
# Aggregation multigrid preconditioner for MatrixFreeLinearSystem
# =============================================================================
#
# Jacobi leaves BiCGStab's iteration count growing with the grid (about 1900 at
# 800 x 200 cells), and on a GPU every iteration costs the same ~0.5 ms of kernel
# launches and reductions whatever the grid size, so the count is the cost. This
# V-cycle holds it at 5-12 from 80 x 20 to 1600 x 400 cells. The V-cycle is
# symmetric, so CGIterativeSolver(g, MatrixFreeLinearSystem) uses it too.
#
# Coarsening merges 2 x 2 cells (unsmoothed, piecewise-constant aggregation) and
# builds each coarse operator as the Galerkin product R*A*P with P the aggregation
# and R = P'. With 2 x 2 aggregates of a 5-point stencil that product is again a
# 5-point stencil, so every level is stored like MatrixFreeLinearSystem (aP and
# positive aE/aW/aN/aS per cell) and every operation is a per-cell kernel: nothing
# leaves the GPU. Galerkin coarsening follows the coefficients wherever K jumps
# (channels next to closed sheet), which a geometric rediscretisation would not.
#
# Rows without neighbour couplings (Dirichlet cells, closed cells) are left out of
# the aggregates (their weight w is 0): the smoother solves them exactly, and
# letting their unit diagonal into a coarse cell would swamp the physical rows'
# K/dx^2. The smoother is red-black Gauss-Seidel, two independent half-sweeps per
# sweep. The coarse correction is scaled by `coarse_scale`: piecewise-constant
# prolongation undershoots smooth errors, and over-correcting compensates
# (Vanek, Brezina & Mandel; Notay's AGMG).

"""
$(TYPEDSIGNATURES)

One level of an [`AggregationMultigrid`](@ref): the 5-point stencil (`aP` and positive `aE`/`aW`/`aN`/`aS`, as in
[`MatrixFreeLinearSystem`](@ref)), the aggregation weights `w` (0 for rows with no neighbour couplings) and the level's
solution and right-hand side.
"""
struct MGLevel{A <: AbstractMatrix}
    aP::A
    aE::A
    aW::A
    aN::A
    aS::A
    w::A
    x::A
    b::A
end

function MGLevel(nx::Int, ny::Int)
    return MGLevel(@zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny))
end

"""
$(TYPEDSIGNATURES)

Matrix-free aggregation multigrid V-cycle over a [`MatrixFreeLinearSystem`](@ref), applied as a Krylov.jl
preconditioner (`M = P, ldiv = true`). Every level lives on the active backend, so it runs on the GPU. See the
note at the top of `multigrid.jl` for the method.

`levels[1]` holds the fine grid's weights and work arrays; its stencil arrays are the system's own. `nsmooth`
red-black Gauss-Seidel sweeps run before and after each coarse correction, `ncoarse` on the coarsest level.
"""
struct AggregationMultigrid{L <: MGLevel, LSy <: MatrixFreeLinearSystem, F <: AbstractFloat}
    levels::Vector{L}
    lsy::LSy
    nsmooth::Int
    ncoarse::Int
    coarse_scale::F
end

"""
$(TYPEDSIGNATURES)

Builds an [`AggregationMultigrid`](@ref) for `mfls`, halving the grid until it has at most `coarsest` cells.
Call [`update_multigrid!`](@ref) after each assembly of `mfls` and before the solve.
"""
function AggregationMultigrid(mfls::MatrixFreeLinearSystem{F}; nsmooth::Int = 2, ncoarse::Int = 20, coarse_scale = 2.0, coarsest::Int = 16) where F
    nsmooth >= 1 || error("nsmooth must be >= 1, got $nsmooth")
    ncoarse >= 1 || error("ncoarse must be >= 1, got $ncoarse")
    nx, ny = size(mfls.aP)
    fine = MGLevel(mfls.aP, mfls.aE, mfls.aW, mfls.aN, mfls.aS, @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny))
    levels = [fine]
    while nx * ny > coarsest && (nx > 1 || ny > 1)
        nx, ny = cld(nx, 2), cld(ny, 2)
        push!(levels, MGLevel(nx, ny))
    end
    return AggregationMultigrid(levels, mfls, nsmooth, ncoarse, F(coarse_scale))
end

# w = 1 for a row coupled to a neighbour, 0 otherwise (Dirichlet and closed cells: identity-like rows the smoother solves exactly)
@parallel_indices (ix, iy) function mg_weight_kernel!(w, aE, aW, aN, aS)
    if ix <= size(w, 1) && iy <= size(w, 2)
        coupled = abs(aE[ix, iy]) + abs(aW[ix, iy]) + abs(aN[ix, iy]) + abs(aS[ix, iy]) > 0
        w[ix, iy] = coupled ? one(eltype(w)) : zero(eltype(w))
    end
    return
end

# Galerkin coarse stencil R*A*P over 2 x 2 aggregates of weighted rows (P[i, I] = w[i] for fine cell i in aggregate I).
# Fine couplings inside an aggregate fold into the coarse diagonal; couplings across an aggregate edge sum into that
# edge's coarse coupling. A coarse cell with no weighted children gets an identity row.
@parallel_indices (I, J) function mg_coarsen_kernel!(aPc, aEc, aWc, aNc, aSc, aP, aE, aW, aN, aS, w)
    nxc, nyc = size(aPc, 1), size(aPc, 2)
    nx, ny = size(aP, 1), size(aP, 2)
    if I <= nxc && J <= nyc
        T = eltype(aPc)
        p, e, ww, n, s = zero(T), zero(T), zero(T), zero(T), zero(T)
        ix0, iy0 = 2I - 1, 2J - 1
        for iy in iy0:min(iy0 + 1, ny), ix in ix0:min(ix0 + 1, nx)
            wi = w[ix, iy]
            if wi != 0
                p += aP[ix, iy]
                # east: inside the aggregate (ix == ix0) folds into the diagonal, across its edge into aEc
                if ix < nx
                    c = aE[ix, iy] * w[ix+1, iy]
                    ix == ix0 ? (p -= c) : (e += c)
                end
                if ix > 1
                    c = aW[ix, iy] * w[ix-1, iy]
                    ix == ix0 + 1 ? (p -= c) : (ww += c)
                end
                if iy < ny
                    c = aN[ix, iy] * w[ix, iy+1]
                    iy == iy0 ? (p -= c) : (n += c)
                end
                if iy > 1
                    c = aS[ix, iy] * w[ix, iy-1]
                    iy == iy0 + 1 ? (p -= c) : (s += c)
                end
            end
        end
        aPc[I, J] = p == 0 ? one(T) : p
        aEc[I, J] = I < nxc ? e : zero(T)
        aWc[I, J] = I > 1 ? ww : zero(T)
        aNc[I, J] = J < nyc ? n : zero(T)
        aSc[I, J] = J > 1 ? s : zero(T)
    end
    return
end

"""
$(TYPEDSIGNATURES)

Rebuilds every coarse level of `P` from its system's current coefficients (two kernels per level).
"""
function update_multigrid!(P::AggregationMultigrid)
    L = P.levels
    for l in eachindex(L)
        f = L[l]
        @parallel (1:size(f.w, 1), 1:size(f.w, 2)) mg_weight_kernel!(f.w, f.aE, f.aW, f.aN, f.aS)
        if l < length(L)
            c = L[l+1]
            @parallel (1:size(c.aP, 1), 1:size(c.aP, 2)) mg_coarsen_kernel!(c.aP, c.aE, c.aW, c.aN, c.aS, f.aP, f.aE, f.aW, f.aN, f.aS, f.w)
        end
    end
    return P
end

refresh_precond!(P::AggregationMultigrid, mfls::MatrixFreeLinearSystem) = update_multigrid!(P) # BiCGStabIterativeSolver's refresh, before each solve

function _cg_precond!(ls::CGIterativeSolver{<:MatrixFreeLinearSystem, WS, V, <:AggregationMultigrid}) where {WS, V <: AbstractVector}
    update_multigrid!(ls.precond) # CGIterativeSolver's refresh, before each solve
    return ls.precond
end

# One red-black Gauss-Seidel half-sweep: updates the cells with (ix + iy) % 2 == color, whose neighbours all have
# the other colour, so the update is in place and order-free.
@parallel_indices (ix, iy) function mg_rbgs_kernel!(x, b, aP, aE, aW, aN, aS, color)
    nx, ny = size(x, 1), size(x, 2)
    if ix <= nx && iy <= ny && (ix + iy) % 2 == color
        r = b[ix, iy]
        ix < nx && (r += aE[ix, iy] * x[ix+1, iy])
        ix > 1  && (r += aW[ix, iy] * x[ix-1, iy])
        iy < ny && (r += aN[ix, iy] * x[ix, iy+1])
        iy > 1  && (r += aS[ix, iy] * x[ix, iy-1])
        x[ix, iy] = r / aP[ix, iy]
    end
    return
end

# Restriction of the residual b - A*x to the coarse right-hand side (bc = R*(b - A*x)), zeroing the coarse solution
@parallel_indices (I, J) function mg_restrict_kernel!(bc, xc, b, x, aP, aE, aW, aN, aS, w)
    nx, ny = size(x, 1), size(x, 2)
    if I <= size(bc, 1) && J <= size(bc, 2)
        acc = zero(eltype(bc))
        for iy in 2J-1:min(2J, ny), ix in 2I-1:min(2I, nx)
            if w[ix, iy] != 0
                r = b[ix, iy] - aP[ix, iy] * x[ix, iy]
                ix < nx && (r += aE[ix, iy] * x[ix+1, iy])
                ix > 1  && (r += aW[ix, iy] * x[ix-1, iy])
                iy < ny && (r += aN[ix, iy] * x[ix, iy+1])
                iy > 1  && (r += aS[ix, iy] * x[ix, iy-1])
                acc += r
            end
        end
        bc[I, J] = acc
        xc[I, J] = zero(eltype(xc))
    end
    return
end

# Prolongation of the coarse correction: x += scale * P*xc
@parallel_indices (ix, iy) function mg_prolong_kernel!(x, xc, w, scale)
    if ix <= size(x, 1) && iy <= size(x, 2)
        x[ix, iy] += scale * w[ix, iy] * xc[cld(ix, 2), cld(iy, 2)]
    end
    return
end

function mg_smooth!(lv::MGLevel, x, b, nsweeps::Int, first::Int)
    rng = (1:size(x, 1), 1:size(x, 2))
    for _ in 1:nsweeps
        @parallel rng mg_rbgs_kernel!(x, b, lv.aP, lv.aE, lv.aW, lv.aN, lv.aS, first)
        @parallel rng mg_rbgs_kernel!(x, b, lv.aP, lv.aE, lv.aW, lv.aN, lv.aS, 1 - first)
    end
    return x
end

# V-cycle from x = 0 on level l: x approximates A_l \ b
function mg_vcycle!(P::AggregationMultigrid, l::Int, x, b)
    lv = P.levels[l]
    if l == length(P.levels)
        # colours 0,1 then 1,0: a palindromic sweep sequence keeps the V-cycle symmetric for CGIterativeSolver
        mg_smooth!(lv, x, b, cld(P.ncoarse, 2), 0)
        mg_smooth!(lv, x, b, fld(P.ncoarse, 2), 1)
        return x
    end
    c = P.levels[l+1]
    mg_smooth!(lv, x, b, P.nsmooth, 0)
    @parallel (1:size(c.b, 1), 1:size(c.b, 2)) mg_restrict_kernel!(c.b, c.x, b, x, lv.aP, lv.aE, lv.aW, lv.aN, lv.aS, lv.w)
    mg_vcycle!(P, l + 1, c.x, c.b)
    @parallel (1:size(x, 1), 1:size(x, 2)) mg_prolong_kernel!(x, c.x, lv.w, P.coarse_scale)
    mg_smooth!(lv, x, b, P.nsmooth, 1)
    return x
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::AggregationMultigrid, r::AbstractVector)
    nx, ny = size(P.lsy.aP)
    x = reshape(y, nx, ny)
    fill!(x, 0)
    mg_vcycle!(P, 1, x, reshape(r, nx, ny))
    return y
end
