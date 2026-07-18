abstract type AbstractLinearSolver end
abstract type AbstractDirectSolver <: AbstractLinearSolver end
abstract type AbstractIterativeSolver <: AbstractLinearSolver end

struct LUSolver{F <: AbstractFloat, FACT} <: AbstractDirectSolver 
    M::SparseMatrixCSC{F, Int}
    fact::FACT
    rhs::Vector{F}
    idxP::Matrix{Int}
    idxE::Matrix{Int}
    idxW::Matrix{Int}
    idxN::Matrix{Int}
    idxS::Matrix{Int}
end

function LUSolver(g::Grid{F}) where F

    # SparseArrays' sparse lu/UMFPACK has no GPU path at all, on any backend
    # -- checking != "Threads" (rather than == "Metal") means this stays
    # correct if a CUDA/AMDGPU backend is ever added, with nothing here to
    # remember to update. Fail here, at construction time, rather than let a
    # GPU-resident array reach it deeper in the solve (same "fail fast in the
    # constructor" idiom as Simulation's FT/floattype mismatch check).
    backend != "Threads" && error("LUSolver is CPU-only (SparseArrays/UMFPACK has no GPU path); choose an iterative solver under the $backend backend.")

    # Create a sparse matrix M representing the 5-point stencil for the 2D Laplacian operator on a grid of size nx by ny. The matrix is constructed in a way that each grid point corresponds to a row in the matrix, and the non-zero entries correspond to the neighboring points (north, south, east, west) and the point itself (center).
    I, J, V = Int[], Int[], F[]

    # Loop over each grid point (i, j) to fill the sparse matrix M with the appropriate coefficients for the 5-point stencil. The diagonal entry corresponds to the center point, and the off-diagonal entries correspond to the neighboring points. The values are set to zero for the neighbors, and a placeholder value of one is used for the diagonal to ensure that the matrix is non-singular for LU factorization.
    for j in 1:g.ny
        @inbounds for i in 1:g.nx

            row = i + (j - 1) * g.nx

            push!(I, row); push!(J, row); push!(V, one(F)) # diagonal - placeholder value 'one' to get nonsingular lu factorization
            i > 1  && (push!(I, row); push!(J, row - 1);  push!(V, zero(F)))  # west
            i < g.nx && (push!(I, row); push!(J, row + 1);  push!(V, zero(F)))  # east
            j > 1  && (push!(I, row); push!(J, row - g.nx); push!(V, zero(F)))  # south
            j < g.ny && (push!(I, row); push!(J, row + g.nx); push!(V, zero(F)))  # north

        end
    end

    # Create the sparse matrix M and perform LU factorization. The sparse matrix is constructed using the row indices (I), column indices (J), and values (V) collected in the previous loop. The LU factorization is performed on the sparse matrix M to prepare for solving linear systems later. The right-hand side vector (rhs) is initialized to zeros, and index arrays (idxP, idxE, idxW, idxN, idxS) are created to map grid points to their corresponding indices in the sparse matrix representation.
    M = sparse(I, J, V, g.nx * g.ny, g.nx * g.ny)
    fact = lu(M)
    rhs = zeros(F, g.nx * g.ny)

    # Create index arrays to map grid points to their corresponding indices in the sparse matrix representation. These arrays will be used later to efficiently access the values in the sparse matrix based on the grid coordinates. The idx_map dictionary is used to store the mapping from (row, col) pairs to their corresponding index in the sparse matrix representation. The loop iterates over each grid point and fills the index arrays with the appropriate indices for the center point (idxP) and its neighboring points (idxE, idxW, idxN, idxS).
    idxP = zeros(Int, g.nx, g.ny)
    idxE = zeros(Int, g.nx, g.ny)
    idxW = zeros(Int, g.nx, g.ny)
    idxN = zeros(Int, g.nx, g.ny)
    idxS = zeros(Int, g.nx, g.ny)

    # Create a mapping from (row, col) pairs to their corresponding index in the sparse matrix representation. This mapping allows for quick access to the indices of non-zero entries in the sparse matrix based on their (row, col) coordinates. The idx_map dictionary is used to store this mapping, where the key is a tuple (rows[k], col) representing the (row, col) pair, and the value is k, which is the index in the sparse matrix representation where this non-zero entry is stored. The loop iterates over each column of the sparse matrix and fills the idx_map dictionary with the appropriate mappings for all non-zero entries.
    idx_map = Dict{Tuple{Int, Int}, Int}()
    rows = rowvals(M) # rowvals(M) returns the row indices of the non-zero entries in the sparse matrix M, in the same order as they appear in the nzval array (the array of non-zero values). This is used to map the (row, col) pairs to their corresponding index in the sparse matrix representation.
    for col in 1:g.nx*g.ny
        @inbounds for k in nzrange(M, col) # nzrange(M, col) returns the range of indices in the rowvals and nzval arrays that correspond to the non-zero entries in column col of the sparse matrix M
            idx_map[(rows[k], col)] = k # This line creates a mapping from the (row, col) pair to the index k in the sparse matrix representation. The key is a tuple (rows[k], col), where rows[k] gives the row index of the non-zero entry, and col is the current column being processed. The value is k, which is the index in the sparse matrix representation where this non-zero entry is stored. This mapping allows for quick access to the indices of non-zero entries in the sparse matrix based on their (row, col) coordinates.
        end
    end

    # Fill the index arrays with the appropriate indices for the center point (idxP) and its neighboring points (idxE, idxW, idxN, idxS). The loop iterates over each grid point (i, j) and uses the idx_map dictionary to look up the corresponding indices in the sparse matrix representation. The idxP array stores the index of the center point, while the idxE, idxW, idxN, and idxS arrays store the indices of the east, west, north, and south neighbors, respectively. The conditional checks ensure that only valid neighbor indices are assigned (i.e., not going out of bounds of the grid).
    for j in 1:g.ny
        @inbounds for i in 1:g.nx
            row = i + (j - 1) * g.nx
            idxP[i, j] = idx_map[(row, row)]
            i > 1  && (idxW[i, j] = idx_map[(row, row - 1)]) # where the (i, j)'s west neighbor's value is stored in the sparse matrix
            i < g.nx && (idxE[i, j] = idx_map[(row, row + 1)])
            j > 1  && (idxS[i, j] = idx_map[(row, row - g.nx)])
            j < g.ny && (idxN[i, j] = idx_map[(row, row + g.nx)])
        end
    end
    
    return LUSolver(M, fact, rhs, idxP, idxE, idxW, idxN, idxS)

end

# mask is passed first so @parallel infers the (ix,iy) launch range from its
# shape (nx,ny) -- nzval/rhs are flat length-(nx*ny) Vectors, and using one of
# those as the first arg would infer a 1D launch instead.
@parallel_indices (ix, iy) function compute_linear_system_kernel!(mask, nzval, rhs, idxP, idxE, idxW, idxN, idxS, zb, h, K, A_visc, N, b, mdot, beta, abs_ub, ieb, dx2, dy2, p_atm, rho_w, rho_i, ggrav, n, n_minus_1, kfs, mi)

    nx, ny = size(mask, 1), size(mask, 2)

    if ix <= nx && iy <= ny

        row = ix + (iy - 1) * nx # linear index of 2D domain with column major order i.e. rows change faster than columns

        m = mask[ix, iy] # mask specifies if a cell is grounded ice, land, ocean, or other basin

        if m == OCEAN # Dirichlet BC

            # Hydrostatic ocean pressure at the bed: zb is elevation relative to sea
            # level (positive up), so a marine bed at zb < 0 sits at depth -zb below
            # sea level. pw = p_atm + rho_w*g*(-zb) = p_atm - rho_w*g*zb.
            # min(zb, 0) clamps to p_atm if this OCEAN-masked cell's bed happens to
            # be above sea level, instead of producing a sub-atmospheric pressure.
            nzval[idxP[ix, iy]] = 1
            pw_bc = p_atm - rho_w * ggrav * min(zb[ix, iy], zero(eltype(zb)))
            rhs[row] = pw_bc / (rho_w * ggrav) + zb[ix, iy] # from hydraulic head h = (pw / (rhow * g)) + zb

        elseif m == LAND # Dirichlet BC

            # pw = p_atm -> h = p_atm/(rho_w*g) + zb
            nzval[idxP[ix, iy]] = 1
            rhs[row] = p_atm / (rho_w * ggrav) + zb[ix, iy]

        elseif m == OTHER_BASIN # Dirichlet BC

            # Not part of this domain's solve: frozen/inert row. h is held at its initial value; via valid_x/valid_y every gradient/
            # flux/melt - in compute_dhdx!, compute_dhdy!, compute_dpwdx!, compute_dpwdy! -
            # computation touching this cell from a GROUNDED neighbour is zeroed, so this frozen value never leaks in.
            nzval[idxP[ix, iy]] = 1
            rhs[row] = h[ix, iy]

        else
            # m == GROUNDED: dynamic hydrology.

            # A face contributes only if the neighbour exists AND is not OTHER_BASIN; both cases (no
            # neighbour / unsolved basin) reduce to a natural zero-flux (Neumann) condition on that face.
            aE = (ix < nx && mask[ix+1, iy] != OTHER_BASIN) ? compute_K_face(kfs, K, ix, iy, ix+1, iy) / dx2 : zero(dx2)
            aW = (ix > 1  && mask[ix-1, iy] != OTHER_BASIN) ? compute_K_face(kfs, K, ix, iy, ix-1, iy) / dx2 : zero(dx2)
            aN = (iy < ny && mask[ix, iy+1] != OTHER_BASIN) ? compute_K_face(kfs, K, ix, iy, ix, iy+1) / dy2 : zero(dy2)
            aS = (iy > 1  && mask[ix, iy-1] != OTHER_BASIN) ? compute_K_face(kfs, K, ix, iy, ix, iy-1) / dy2 : zero(dy2)

            # Last aP term comes from Newton linearization of the creep closing term - appendix of doi: 10.1017/jog.2018.59.
            # pow(..., n_minus_1) (n_minus_1 = p.n_minus_1_exp, canonicalized once at
            # ModelParameters construction) rather than abs(N)^(n-1): see
            # model_parameters.jl's pow/canonical_exponent note -- n-1 is an Int for
            # the standard Glen's-law n=3, hitting the fast power-by-squaring path.
            aP = (aE + aW + aN + aS) + n * rho_w * ggrav * A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * b[ix, iy]

            # Update the non-zero values of the M sparse matrix
            nzval[idxP[ix, iy]] += aP
            ix < nx && (nzval[idxE[ix, iy]] -= aE)
            ix > 1  && (nzval[idxW[ix, iy]] -= aW)
            iy < ny && (nzval[idxN[ix, iy]] -= aN)
            iy > 1  && (nzval[idxS[ix, iy]] -= aS)

            # Update the rhs vector
            rhs[row] = mdot[ix, iy] * (1 / rho_w - 1 / rho_i) -
                        beta[ix, iy] * abs_ub[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * b[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * (n * rho_w * ggrav * h[ix, iy]) * b[ix, iy] + # term from Newton linearization of the creep closing term
                        compute_ieb!(mi, ieb, ix, iy)
        end

    end

    return
end

function compute_linear_system!(ls::LUSolver, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, mi::AbstractMeltInput)

    # Unpack the solver's preallocated matrix/rhs/index-map workspace (see LUSolver's docstring)
    rhs = ls.rhs
    nzval = ls.M.nzval
    idxP, idxE, idxW, idxN, idxS = ls.idxP, ls.idxE, ls.idxW, ls.idxN, ls.idxS

    # We have a new linear system so we refresh the M and rhs that we will now re-build with new values
    fill!(nzval, 0)
    fill!(rhs, 0)

    @parallel compute_linear_system_kernel!(s.mask, nzval, rhs, idxP, idxE, idxW, idxN, idxS, s.zb, s.h, s.K, s.A_visc, s.N, s.b, s.mdot, s.beta, s.abs_ub, s.ieb, g.dx2, g.dy2, p.p_atm, p.rho_w, p.rho_i, p.g, p.n, p.n_minus_1_exp, kfs, mi)

    return ls
end

function solve_linear_system!(ls::LUSolver, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, mi::AbstractMeltInput)

    compute_linear_system!(ls, s, g, p, kfs, mi) # prepare the new linear system sparse matrix M and rhs

    lu!(ls.fact, ls.M) # computes the LU factorization of the matrix ls.M in-place, storing the result in the preallocated factorization object ls.fact

    ldiv!(s.h_vec, ls.fact, ls.rhs) # solve for the new h based on the new LU decomposition of the sparse M matrix we have computed above

    s.h .= reshape(s.h_vec, g.nx, g.ny) # update h

end
