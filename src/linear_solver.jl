# Known hydraulic head at a Dirichlet (OCEAN/LAND) cell -- shared by that
# cell's own row (h_j = g_j) and by any GROUNDED neighbour that needs to fold
# g_j into its rhs instead of coupling to it in the matrix (see the symmetric
# elimination comment in update_SALS_elliptic_kernel!/update_MFLS_elliptic_kernel!).
"""
$(TYPEDSIGNATURES)

Known hydraulic head at a Dirichlet (`OCEAN`/`LAND`) cell -- shared by that cell's own row
(`h_j = g_j`) and by any `GROUNDED` neighbour that needs to fold `g_j` into its rhs instead of
coupling to it in the matrix (see the symmetric-elimination note on [`update_SALS_elliptic!`](@ref)/
[`update_MFLS_elliptic!`](@ref)).
"""
@inline function dirichlet_head(m, zb_ij, p_atm, rho_w, rho_sw, ggrav)
    if m == OCEAN
        # Hydrostatic ocean pressure at the bed: zb is elevation relative to sea
        # level (positive up), so a marine bed at zb < 0 sits at depth -zb below
        # sea level. pw = p_atm + rho_sw*g*(-zb) = p_atm - rho_sw*g*zb, using
        # rho_sw (seawater) rather than rho_w (subglacial water) since this is
        # the ocean's own hydrostatic pressure, not the drainage system's.
        # min(zb, 0) clamps to p_atm if this OCEAN-masked cell's bed happens to
        # be above sea level, instead of producing a sub-atmospheric pressure.
        pw_bc = p_atm - rho_sw * ggrav * min(zb_ij, zero(zb_ij))
        return pw_bc / (rho_w * ggrav) + zb_ij # h = (pw / (rho_w * g)) + zb -- head is always defined w.r.t. rho_w, the drainage system's own water density
    else # m == LAND: pw = p_atm -> h = p_atm/(rho_w*g) + zb
        return p_atm / (rho_w * ggrav) + zb_ij
    end
end

"""
$(TYPEDSIGNATURES)

How the linearized elliptic equation for `h` (one per Picard iteration) is solved -- multiple
dispatch on the concrete subtype picks a direct ([`AbstractDirectSolver`](@ref)) or iterative
([`AbstractIterativeSolver`](@ref)) method, via [`solve_elliptic_linear_system!`](@ref).
"""
abstract type AbstractLinearSolver end

"""
$(TYPEDSIGNATURES)

A direct linear solver (currently just [`CholeskyDirectSolver`](@ref)).
"""
abstract type AbstractDirectSolver <: AbstractLinearSolver end

"""
$(TYPEDSIGNATURES)

An iterative linear solver (currently just [`CGIterativeSolver`](@ref)).
"""
abstract type AbstractIterativeSolver <: AbstractLinearSolver end

"""
$(TYPEDSIGNATURES)

Whether `b`'s evolution includes the new channel-wall diffusion term (`∇·(D∇b)`, Felden et al.
2023 -- SUHMO -- Eqs. 9-11, https://doi.org/10.5194/gmd-16-407-2023) -- multiple dispatch on the
concrete subtype ([`NoDiffusion`](@ref)/[`WithDiffusion`](@ref)) picks whether the diffusion
coefficient `D` (`s.D_x`/`s.D_y`, `melt_rate.jl`'s `compute_D!`) is refreshed every Picard
iteration and whether `b`'s own timestep update solves the coupled diffusion system at all, same
"dispatch on a type decided once outside the hot loop" idiom as [`AbstractOpenBySlidingScheme`](@ref)/
[`AbstractCreepLengthScheme`](@ref) (`gap_height.jl`). Defined here (not in `gap_height.jl`, where
those two live) because [`WithDiffusion`](@ref) bundles its own [`AbstractLinearSolver`](@ref) --
`b`'s own diffusion operator needs a second, independent linear solve alongside `h`'s (different
sparsity coefficients: `1` plus face diffusivities vs. `h`'s Newton-linearized creep-closure
reaction term), not a shared one -- and this file (`linear_solver.jl`) is included well before
`elliptic_solver.jl`, whose `refresh_head_dependents!`/`Picard_iteration!` need
`AbstractDiffusionScheme` in scope to type-annotate `ds`.
"""
abstract type AbstractDiffusionScheme end

"""
$(TYPEDSIGNATURES)

Today's behavior: `b`'s evolution has no diffusion term, `D` is never computed (its call is
skipped entirely, not just multiplied by a zero prefactor, same "not just multiplied by zero"
idiom as [`NoOpenBySliding`](@ref)).
"""
struct NoDiffusion <: AbstractDiffusionScheme end

"""
$(TYPEDSIGNATURES)

Includes the channel-wall diffusion term in `b`'s evolution (and the matching `-∇·(D∇b)` source
term in `h`'s elliptic/parabolic RHS): `D` is refreshed every Picard iteration from the current
`q`/`∇h`/`∇P_w` (Eq. 10), and `b`'s own timestep update ([`compute_b!`](@ref)) solves the resulting
coupled linear system using `ls` -- a *second*, independent [`AbstractLinearSolver`](@ref) instance
from whatever solves `h` (`CholeskyDirectSolver`/`CGIterativeSolver`, same types, just built on
`b`'s own operator).
"""
struct WithDiffusion{LS <: AbstractLinearSolver} <: AbstractDiffusionScheme
    ls::LS
end

"""
$(TYPEDSIGNATURES)

How the assembled linear system is represented -- multiple dispatch on the concrete subtype picks
between an explicit sparse matrix ([`SparseAssembledLinearSystem`](@ref), CPU-only, needed by
[`CholeskyDirectSolver`](@ref)) and a matrix-free stencil representation
([`MatrixFreeLinearSystem`](@ref), GPU-capable).
"""
abstract type AbstractLinearSystem end

"""
$(TYPEDSIGNATURES)

The linear system as an explicit sparse matrix `M` (5-point stencil) and dense `rhs`, plus index
arrays (`idxP`/`idxE`/`idxW`/`idxN`/`idxS`) mapping each grid cell to where its diagonal/neighbour
coefficients live in `M.nzval` -- built once (the sparsity pattern never changes across Picard
iterations/timesteps) so [`update_SALS_elliptic!`](@ref) only ever rewrites values, not structure. CPU-only
(`SparseArrays`/CHOLMOD have no GPU path); see [`MatrixFreeLinearSystem`](@ref) for the
GPU-capable alternative.
"""
struct SparseAssembledLinearSystem{F <: AbstractFloat} <: AbstractLinearSystem
    M::SparseMatrixCSC{F, Int}
    rhs::Vector{F}
    idxP::Matrix{Int}
    idxE::Matrix{Int}
    idxW::Matrix{Int}
    idxN::Matrix{Int}
    idxS::Matrix{Int}
end

"""
$(TYPEDSIGNATURES)

The linear system as per-cell stencil coefficients (`aP`/`aE`/`aW`/`aN`/`aS`, one value per grid
cell each) with no sparse matrix ever materialized -- the matvec ([`stencil_matvec_kernel!`](@ref)
via [`StencilOperator`](@ref)) applies the stencil directly. GPU-capable (every array lands on the
active backend, see the constructor below); the alternative to
[`SparseAssembledLinearSystem`](@ref) for backends without a sparse-direct-solver path.
"""
struct MatrixFreeLinearSystem{F <: AbstractFloat, A <: AbstractMatrix{F}, V <: AbstractVector{F}} <: AbstractLinearSystem
    aP::A # diagonal coefficient, one per grid cell
    aE::A # east-neighbor coupling (stored positive; the matvec applies the minus sign)
    aW::A # west-neighbor coupling
    aN::A # north-neighbor coupling
    aS::A # south-neighbor coupling
    rhs::V # flat nx*ny rather than nx, ny, so it can be handed to Krylov.jl directly like SparseAssembledLinearSystem's rhs
end

"""
$(TYPEDSIGNATURES)

Builds an empty (all-zero) [`MatrixFreeLinearSystem`](@ref) on grid `g`, with every array
allocated on `g`'s active backend.
"""
function MatrixFreeLinearSystem(g::Grid{F}) where F

    # @zeros/initialize_center_field (not plain zeros/Array) so these land on
    # the active backend (Threads -> Array, Metal -> MtlArray), matching
    # State's fields -- this is what actually makes "matrix-free" GPU-capable:
    # everything the assembly/matvec kernels touch lives on the same device.
    aP = initialize_center_field(g)
    aE = initialize_center_field(g)
    aW = initialize_center_field(g)
    aN = initialize_center_field(g)
    aS = initialize_center_field(g)
    rhs = @zeros(g.nx * g.ny)

    return MatrixFreeLinearSystem(aP, aE, aW, aN, aS, rhs)

end

"""
$(TYPEDSIGNATURES)

Direct solve via sparse Cholesky factorization (`SparseArrays`/CHOLMOD) of the assembled
[`SparseAssembledLinearSystem`](@ref) -- the fastest solver at every grid size benchmarked so far,
but CPU-only. Refactorizes in place each solve, reusing the symbolic
factorization since the sparsity pattern never changes.
"""
struct CholeskyDirectSolver{FACT, SALS <: SparseAssembledLinearSystem, V <: AbstractVector} <: AbstractDirectSolver
    sals::SALS # holds sparse matrix non-zero values, rhs vector, and indices of where the non-zero values of the matrix are
    fact::FACT # going to hold the decomposition factorization of cholesky
    h_vec::V # buffer the `\`-solve result is copied into (vectorized hydraulic head) before reshaping into s.h
end

"""
$(TYPEDSIGNATURES)

Builds a [`SparseAssembledLinearSystem`](@ref) on grid `g`: a 5-point-stencil sparse matrix `M`
(placeholder zero off-diagonal values, filled in properly by [`update_SALS_elliptic!`](@ref) each solve)
and the index arrays needed to address into `M.nzval` by grid coordinate.
"""
function SparseAssembledLinearSystem(g::Grid{F}) where F

    # Create a sparse matrix M representing the 5-point stencil for the 2D Laplacian operator on a grid of size nx by ny. The matrix is constructed in a way that each grid point corresponds to a row in the matrix, and the non-zero entries correspond to the neighboring points (north, south, east, west) and the point itself (center).
    I, J, V = Int[], Int[], F[]

    # Loop over each grid point (i, j) to fill the sparse matrix M with the appropriate coefficients for the 5-point stencil. The diagonal entry corresponds to the center point, and the off-diagonal entries correspond to the neighboring points. The values are set to zero for the neighbors, and a placeholder value of one is used for the diagonal to ensure that the matrix is non-singular for LU factorization.
    for j in 1:g.ny
        @inbounds for i in 1:g.nx

            row = i + (j - 1) * g.nx # this mapping from i, j to the linear index of the sparse matrix determines why we write row-1, row-g.nx, etc here below

            push!(I, row); push!(J, row); push!(V, one(F)) # diagonal - placeholder value 'one' to get a nonsingular factorization
            i > 1  && (push!(I, row); push!(J, row - 1);  push!(V, zero(F)))  # west
            i < g.nx && (push!(I, row); push!(J, row + 1);  push!(V, zero(F)))  # east
            j > 1  && (push!(I, row); push!(J, row - g.nx); push!(V, zero(F)))  # south
            j < g.ny && (push!(I, row); push!(J, row + g.nx); push!(V, zero(F)))  # north

        end
    end

    # Create the sparse matrix M. The sparse matrix is constructed using the row indices (I), column indices (J), and values (V) collected in the previous loop. The right-hand side vector (rhs) is initialized to zeros, and index arrays (idxP, idxE, idxW, idxN, idxS) are created to map grid points to their corresponding indices in the sparse matrix representation.
    M = sparse(I, J, V, g.nx * g.ny, g.nx * g.ny)
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

    return SparseAssembledLinearSystem{F}(M, rhs, idxP, idxE, idxW, idxN, idxS)

end

"""
$(TYPEDSIGNATURES)

Builds a [`CholeskyDirectSolver`](@ref) on grid `g`. CPU (`"Threads"` backend) only -- errors at
construction time (rather than deeper in the solve) under any other backend.
"""
function CholeskyDirectSolver(g::Grid{F}) where F

    # Fail here, at construction time, rather than let a GPU-resident array reach it deeper in the solve
    backend != "Threads" && error("CholeskyDirectSolver is CPU-only (SparseArrays/CHOLMOD has no GPU path); choose an iterative solver under the $backend backend.")

    sals = SparseAssembledLinearSystem(g)
    # Symmetric(...) tells CHOLMOD to read only one triangle -- valid now that
    # sals.M is exactly symmetric (Dirichlet neighbours are eliminated
    # symmetrically, see update_SALS_elliptic_kernel!) and symmetric positive-definite SPD (diffusion + strictly
    # positive diagonal reaction term from the Newton-linearized creep closure).
    fact = cholesky(Symmetric(sals.M)) # CHOLMOD is SuiteSparse's sparse Cholesky factorization library, exposed in Julia through SparseArrays/LinearAlgebra's cholesky function. In linear_solver.jl, cholesky(Symmetric(sals.M)) calls into CHOLMOD to factorize the sparse SPD matrix from the assembled linear system
    h_vec = zeros(F, g.nx * g.ny)

    return CholeskyDirectSolver(sals, fact, h_vec)

end

"""
$(TYPEDSIGNATURES)

The new `-∇·(D∇b)` source term SUHMO's Eq. 11 adds to `h`'s elliptic/parabolic RHS -- substituting
`b`'s own evolution equation (`gap_height.jl`) into mass conservation to eliminate `∂b/∂t` is
exactly what already puts the melt/closure/opening terms into the RHS (see `update_SALS_elliptic_kernel!`'s
own `rhs` assignment below); this is that same substitution, extended to the new spatial term.
Zero under [`NoDiffusion`](@ref) -- dispatched away entirely, not computed and multiplied by zero,
same idiom as [`MeltTerms`](@ref)'s own terms.

Discretized as a standard 5-point finite-volume divergence, `Σ_neighbours D_face*(b_neighbour -
b_here) / dx2_or_dy2` (summed over whichever of the 4 faces are "active"), negated for the RHS's
own `-∇·(D∇b)` (`acc` below is `+∇·(D∇b)`). A face contributes zero-flux (Neumann) when there's no
neighbour in that direction at all (domain edge, mirroring `aE`/`aW`/`aN`/`aS`'s own bounds checks
just below) or when the neighbour is anything other than `GROUNDED`:

  - `OCEAN`/`LAND`: no physically meaningful `b` to diffuse into/from there -- `b` at those cells
    is a bookkeeping placeholder (forced to `0` in `set_initial_conditions!`), not a real gap
    height (same convention `k_face_scheme.jl`'s own note describes for `K`). Reading it directly
    would create a standing artificial drain toward the domain boundary at every margin cell,
    forever -- not something the physics asked for.
  - `OTHER_BASIN`/`FROZEN_BED`: `D` is already exactly zero at these faces automatically (inherited
    from `compute_D_kernel!`'s own zero-cascade through `dhdx`/`dpwdx`/`q_x`, `melt_rate.jl`), so
    this `GROUNDED`-only check is a no-op restatement for them, not new behaviour.

Dispatches on `Val{true}`/`Val{false}` rather than directly on `WithDiffusion`/`NoDiffusion`
(`ds`) itself: `@parallel` infers a kernel's launch range by inspecting every one of its
arguments, requiring each to be an array, scalar, or other `isbits` type -- true of every other
scheme marker passed into these kernels (`Arithmetic()`, `MeltTerms{...}()`, `StandardCreep()`,
all empty structs), but *not* of `WithDiffusion`, whose `ls` field holds a real
[`AbstractLinearSolver`](@ref) (sparse matrices, factorizations, ...). The wrapper functions below
convert `ds` to `Val(ds isa WithDiffusion)` -- an `isbits`, zero-size flag carrying no more than
the one bit `@parallel` and this dispatch actually need -- right before the kernel call, rather
than threading the heavy `ds` itself all the way into the hot loop.
"""
@inline diffusion_source(::Val{false}, D_x, D_y, mask, b, ix, iy, nx, ny, dx2, dy2) = zero(eltype(b))

@inline function diffusion_source(::Val{true}, D_x, D_y, mask, b, ix, iy, nx, ny, dx2, dy2)
    acc = zero(eltype(b))
    if ix < nx && mask[ix+1, iy] == GROUNDED
        acc += D_x[ix+1, iy] * (b[ix+1, iy] - b[ix, iy]) / dx2
    end
    if ix > 1 && mask[ix-1, iy] == GROUNDED
        acc += D_x[ix, iy] * (b[ix-1, iy] - b[ix, iy]) / dx2
    end
    if iy < ny && mask[ix, iy+1] == GROUNDED
        acc += D_y[ix, iy+1] * (b[ix, iy+1] - b[ix, iy]) / dy2
    end
    if iy > 1 && mask[ix, iy-1] == GROUNDED
        acc += D_y[ix, iy] * (b[ix, iy-1] - b[ix, iy]) / dy2
    end
    return -acc
end

# mask is passed first so @parallel infers the (ix,iy) launch range from its
# shape (nx,ny) -- nzval/rhs are flat length-(nx*ny) Vectors, and using one of
# those as the first arg would infer a 1D launch instead.
@parallel_indices (ix, iy) function update_SALS_elliptic_kernel!(mask, nzval, rhs, idxP, idxE, idxW, idxN, idxS, zb, h, K, A_visc, N, lc, mdot, beta, abs_ub, ieb, dx2, dy2, p_atm, rho_w, rho_sw, rho_i, ggrav, n, n_minus_1, kfs, b, D_x, D_y, diffusion_on)

    nx, ny = size(mask, 1), size(mask, 2)

    if ix <= nx && iy <= ny

        row = ix + (iy - 1) * nx # linear index of 2D domain with column major order i.e. rows change faster than columns

        m = mask[ix, iy] # mask specifies if a cell is grounded ice, ocean, land, or other basin

        if m == OCEAN || m == LAND # Dirichlet BC

            nzval[idxP[ix, iy]] = 1 # set the diagonal value of that row to 1 and the rhs to the value of h we want that row - which represents an i, j grid point - to have
            rhs[row] = dirichlet_head(m, zb[ix, iy], p_atm, rho_w, rho_sw, ggrav)

        elseif m == OTHER_BASIN || m == FROZEN_BED # Dirichlet BC

            # Not part of this domain's solve: frozen/inert row. h is held at whatever it currently
            # is (its initial value, unless a FROZEN_BED cell's pw/h was explicitly reset at the
            # moment it froze -- see mask.jl's FROZEN_BED docstring); via valid_x/valid_y every
            # gradient/flux/melt - in compute_dhdx!, compute_dhdy!, compute_dpwdx!, compute_dpwdy! -
            # computation touching this cell from a GROUNDED neighbour is zeroed, so this frozen value never leaks in.
            nzval[idxP[ix, iy]] = 1
            rhs[row] = h[ix, iy] # frozen at whatever value it currently holds

        else
            # m == GROUNDED: dynamic hydrology.

            # A face contributes only if the neighbour exists; no neighbour reduces to a natural
            # zero-flux (Neumann) condition. boundary_K_face handles OTHER_BASIN/FROZEN_BED/OCEAN/LAND neighbours
            # (see k_face_scheme.jl).
            aE = (ix < nx) ? boundary_K_face(kfs, K, mask, ix, iy, ix+1, iy) / dx2 : zero(dx2)
            aW = (ix > 1)  ? boundary_K_face(kfs, K, mask, ix, iy, ix-1, iy) / dx2 : zero(dx2)
            aN = (iy < ny) ? boundary_K_face(kfs, K, mask, ix, iy, ix, iy+1) / dy2 : zero(dy2)
            aS = (iy > 1)  ? boundary_K_face(kfs, K, mask, ix, iy, ix, iy-1) / dy2 : zero(dy2)

            # Last aP term comes from Newton linearization of the creep closing term - appendix of doi: 10.1017/jog.2018.59.
            # pow(..., n_minus_1) (n_minus_1 = p.n_minus_1_exp, canonicalized once at
            # ModelParameters construction) rather than abs(N)^(n-1): see
            # model_parameters.jl's pow/canonical_exponent note -- n-1 is an Int for
            # the standard Glen's-law n=3, hitting the fast power-by-squaring path.
            # `lc` (not `b` directly) is the ice-creep length scale -- lagged like `beta`,
            # see AbstractCreepLengthScheme (gap_height.jl); equals `b` under StandardCreep.
            aP = (aE + aW + aN + aS) + n * rho_w * ggrav * A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * lc[ix, iy]

            # Update the non-zero values of the M sparse matrix. A GROUNDED
            # neighbour couples symmetrically (handled below in the else
            # branch). An OCEAN/LAND neighbour's head is already known, so
            # rather than coupling to it through a matrix entry that its own
            # row (a bare identity row, see above) would never reciprocate --
            # which is what breaks symmetry -- its contribution is folded (added)
            # straight into rhs instead, matching the symmetric elimination
            # of Dirichlet dofs (fold known value into the coupled rows' rhs,
            # zero both the row and the column for that dof).
            nzval[idxP[ix, iy]] += aP
            dirichlet_rhs = zero(eltype(rhs))
            if ix < nx
                mE = mask[ix+1, iy]
                (mE == OCEAN || mE == LAND) ? (dirichlet_rhs += aE * dirichlet_head(mE, zb[ix+1, iy], p_atm, rho_w, rho_sw, ggrav)) : (nzval[idxE[ix, iy]] -= aE) # so again first case here is adding the known value to the rhs and the second case is adding a non-diagonal element to the row
            end
            if ix > 1
                mW = mask[ix-1, iy]
                (mW == OCEAN || mW == LAND) ? (dirichlet_rhs += aW * dirichlet_head(mW, zb[ix-1, iy], p_atm, rho_w, rho_sw, ggrav)) : (nzval[idxW[ix, iy]] -= aW)
            end
            if iy < ny
                mN = mask[ix, iy+1]
                (mN == OCEAN || mN == LAND) ? (dirichlet_rhs += aN * dirichlet_head(mN, zb[ix, iy+1], p_atm, rho_w, rho_sw, ggrav)) : (nzval[idxN[ix, iy]] -= aN)
            end
            if iy > 1
                mS = mask[ix, iy-1]
                (mS == OCEAN || mS == LAND) ? (dirichlet_rhs += aS * dirichlet_head(mS, zb[ix, iy-1], p_atm, rho_w, rho_sw, ggrav)) : (nzval[idxS[ix, iy]] -= aS)
            end

            # Update the rhs vector
            rhs[row] = mdot[ix, iy] * (1 / rho_w - 1 / rho_i) -
                        beta[ix, iy] * abs_ub[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * lc[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * (n * rho_w * ggrav * h[ix, iy]) * lc[ix, iy] + # term from Newton linearization of the creep closing term
                        ieb[ix, iy] +
                        diffusion_source(diffusion_on, D_x, D_y, mask, b, ix, iy, nx, ny, dx2, dy2) + # -div(D*grad(b)) (SUHMO Eq. 11), zero under NoDiffusion
                        dirichlet_rhs
        end

    end

    return
end

"""
$(TYPEDSIGNATURES)

Rebuilds `sals.M`/`sals.rhs` in place from the current `s`/`p` (mask, head, transmissivity, ...):
`GROUNDED` cells get the diffusion + Newton-linearized creep-closure stencil, `OCEAN`/`LAND` cells
get a Dirichlet row, `OTHER_BASIN`/`FROZEN_BED` cells get a frozen (`h` held at its current value)
row. `M` stays exactly symmetric: a `GROUNDED` cell's Dirichlet neighbours are eliminated by
folding their known head into `rhs` rather than left as a one-sided matrix coupling.
"""
function update_SALS_elliptic!(sals::SparseAssembledLinearSystem, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, ds::AbstractDiffusionScheme = NoDiffusion())

    # Unpack the solver's preallocated matrix/rhs/index-map workspace (see CholeskyDirectSolver's docstring)
    rhs = sals.rhs
    nzval = sals.M.nzval
    idxP, idxE, idxW, idxN, idxS = sals.idxP, sals.idxE, sals.idxW, sals.idxN, sals.idxS

    # We have a new linear system so we refresh the M and rhs that we will now re-build with new values
    fill!(nzval, 0)
    fill!(rhs, 0)

    @parallel update_SALS_elliptic_kernel!(s.mask, nzval, rhs, idxP, idxE, idxW, idxN, idxS, s.zb, s.h, s.K, s.A_visc, s.N, s.lc, s.mdot, s.beta, s.abs_ub, s.ieb, g.dx2, g.dy2, p.p_atm, p.rho_w, p.rho_sw, p.rho_i, p.g, p.n, p.n_minus_1_exp, kfs, s.b, s.D_x, s.D_y, Val(ds isa WithDiffusion))

    return

end

# Sibling of update_SALS_elliptic_kernel! for the parabolic head scheme (p.e_v != 0,
# see ParabolicHeadScheme in simulation.jl): backward-Euler discretization of
# Sommers et al. 2018 Eq. 13's ∂(e_v(h-zb))/∂t storage term -- zb is static,
# so this is just e_v*(h-h_old)/dt, giving the GROUNDED row an extra e_v/dt
# on the diagonal and e_v/dt*h_old on the rhs, ON TOP OF the same Newton
# linearization of the creep-closure term update_SALS_elliptic_kernel! has (aP
# gets n*rho_w*g*A_visc*|N|^(n-1)*b too, rhs gets the matching Newton
# correction term, evaluated at `h` -- the CURRENT Picard-sub-iterate, the
# correct linearization point). `h_old` is deliberately a SEPARATE argument
# from `h`: it must stay fixed at the value from the *start of the real
# timestep* across every Parabolic_loop! sub-iteration, whereas `h` is
# whatever the current sub-iterate is. Passing `s.h` for both (as if
# "current" and "old" were the same array) is exactly what parabolic_solver!
# does correctly for its own single, non-iterated call (there they genuinely
# coincide, since nothing has overwritten s.h yet) -- but reusing that pattern
# inside an outer loop silently turns "backward-Euler from the timestep start"
# into "backward-Euler from the previous Picard sub-iterate" from the second
# sub-iteration on, which is a different (and not generally convergent)
# equation. Confirmed to matter empirically, not just in theory: before this
# was split out, Parabolic_loop! failed to converge at all on stiff cells
# (thin, marine-grounded ice near flotation) on the real Greenland grid --
# 500/500 iterations every step, N<0 growing without bound, even with the
# Newton term above already in place. K/A_visc/b/mdot/beta/abs_ub are still
# lagged at the current iterate (only the creep term's own N-dependence is
# Newton-corrected, same as the elliptic kernel) -- one full Parabolic_loop!
# call still drives all of them to self-consistency across iterations, same
# as Picard_loop! does.
@parallel_indices (ix, iy) function update_SALS_parabolic_kernel!(mask, nzval, rhs, idxP, idxE, idxW, idxN, idxS, zb, h, h_old, K, A_visc, N, lc, mdot, beta, abs_ub, ieb, dx2, dy2, p_atm, rho_w, rho_sw, rho_i, ggrav, n, n_minus_1, kfs, e_v, dt, b, D_x, D_y, diffusion_on)

    nx, ny = size(mask, 1), size(mask, 2)

    if ix <= nx && iy <= ny

        row = ix + (iy - 1) * nx

        m = mask[ix, iy]

        if m == OCEAN || m == LAND # Dirichlet BC

            nzval[idxP[ix, iy]] = 1
            rhs[row] = dirichlet_head(m, zb[ix, iy], p_atm, rho_w, rho_sw, ggrav)

        elseif m == OTHER_BASIN || m == FROZEN_BED # Dirichlet BC

            nzval[idxP[ix, iy]] = 1
            rhs[row] = h[ix, iy]

        else
            # m == GROUNDED: dynamic hydrology.

            aE = (ix < nx) ? boundary_K_face(kfs, K, mask, ix, iy, ix+1, iy) / dx2 : zero(dx2)
            aW = (ix > 1)  ? boundary_K_face(kfs, K, mask, ix, iy, ix-1, iy) / dx2 : zero(dx2)
            aN = (iy < ny) ? boundary_K_face(kfs, K, mask, ix, iy, ix, iy+1) / dy2 : zero(dy2)
            aS = (iy > 1)  ? boundary_K_face(kfs, K, mask, ix, iy, ix, iy-1) / dy2 : zero(dy2)

            aP = (aE + aW + aN + aS) + e_v / dt + n * rho_w * ggrav * A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * lc[ix, iy] # diffusion + backward-Euler englacial storage reaction term + Newton-linearized creep closure (same term as update_SALS_elliptic_kernel!'s aP)

            nzval[idxP[ix, iy]] += aP
            dirichlet_rhs = zero(eltype(rhs))
            if ix < nx
                mE = mask[ix+1, iy]
                (mE == OCEAN || mE == LAND) ? (dirichlet_rhs += aE * dirichlet_head(mE, zb[ix+1, iy], p_atm, rho_w, rho_sw, ggrav)) : (nzval[idxE[ix, iy]] -= aE)
            end
            if ix > 1
                mW = mask[ix-1, iy]
                (mW == OCEAN || mW == LAND) ? (dirichlet_rhs += aW * dirichlet_head(mW, zb[ix-1, iy], p_atm, rho_w, rho_sw, ggrav)) : (nzval[idxW[ix, iy]] -= aW)
            end
            if iy < ny
                mN = mask[ix, iy+1]
                (mN == OCEAN || mN == LAND) ? (dirichlet_rhs += aN * dirichlet_head(mN, zb[ix, iy+1], p_atm, rho_w, rho_sw, ggrav)) : (nzval[idxN[ix, iy]] -= aN)
            end
            if iy > 1
                mS = mask[ix, iy-1]
                (mS == OCEAN || mS == LAND) ? (dirichlet_rhs += aS * dirichlet_head(mS, zb[ix, iy-1], p_atm, rho_w, rho_sw, ggrav)) : (nzval[idxS[ix, iy]] -= aS)
            end

            # rhs: source terms lagged at the current iterate, the Newton correction for the
            # creep term's own N-dependence (linearized at `h`, same as
            # update_SALS_elliptic_kernel!'s rhs), and the e_v/dt*h_old storage term (h_old, NOT
            # h -- see this kernel's own docstring note above)
            rhs[row] = mdot[ix, iy] * (1 / rho_w - 1 / rho_i) -
                        beta[ix, iy] * abs_ub[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * lc[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * (n * rho_w * ggrav * h[ix, iy]) * lc[ix, iy] + # term from Newton linearization of the creep closing term
                        ieb[ix, iy] +
                        diffusion_source(diffusion_on, D_x, D_y, mask, b, ix, iy, nx, ny, dx2, dy2) + # -div(D*grad(b)) (SUHMO Eq. 11), zero under NoDiffusion
                        (e_v / dt) * h_old[ix, iy] +
                        dirichlet_rhs
        end

    end

    return
end

"""
$(TYPEDSIGNATURES)

Rebuilds `sals.M`/`sals.rhs` in place for the parabolic head scheme ([`update_SALS_parabolic_kernel!`](@ref)) --
the `p.e_v != 0` counterpart of [`update_SALS_elliptic!`](@ref): same diffusion stencil,
Dirichlet/frozen rows, and Newton-linearized creep-closure term, plus a backward-Euler englacial
storage reaction term (`p.e_v/dt*h_old`) on top -- every other nonlinear coefficient (`K`,
`A_visc`, `b`, `mdot`, `beta`, `abs_ub`) is still lagged at the current iterate, refreshed across
iterations by [`Parabolic_loop!`](@ref) (`parabolic_solver.jl`) rather than within this one
assembly. `h_old` (the head at the *start of the real timestep*, fixed across every
[`Parabolic_loop!`](@ref) sub-iteration) is taken as an explicit argument, separate from `s.h`
(the current Picard sub-iterate) -- see [`update_SALS_parabolic_kernel!`](@ref)'s own docstring
note for why conflating the two breaks convergence.
"""
function update_SALS_parabolic!(sals::SparseAssembledLinearSystem, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, dt, h_old, ds::AbstractDiffusionScheme = NoDiffusion())

    rhs = sals.rhs
    nzval = sals.M.nzval
    idxP, idxE, idxW, idxN, idxS = sals.idxP, sals.idxE, sals.idxW, sals.idxN, sals.idxS

    fill!(nzval, 0)
    fill!(rhs, 0)

    @parallel update_SALS_parabolic_kernel!(s.mask, nzval, rhs, idxP, idxE, idxW, idxN, idxS, s.zb, s.h, h_old, s.K, s.A_visc, s.N, s.lc, s.mdot, s.beta, s.abs_ub, s.ieb, g.dx2, g.dy2, p.p_atm, p.rho_w, p.rho_sw, p.rho_i, p.g, p.n, p.n_minus_1_exp, kfs, p.e_v, dt, s.b, s.D_x, s.D_y, Val(ds isa WithDiffusion))

    return

end

# Sibling of update_SALS_elliptic_kernel! for the matrix-free representation: fills
# per-cell coefficients directly (aP/aE/aW/aN/aS), no nzval/idx* scatter
# needed since there's no sparse matrix to address into. aE/aW/aN/aS are
# stored as the raw positive face conductances -- stencil_matvec_kernel!
# below applies the minus sign when it uses them, matching the sign
# convention update_SALS_elliptic_kernel! bakes directly into nzval.
@parallel_indices (ix, iy) function update_MFLS_elliptic_kernel!(mask, aP, aE, aW, aN, aS, rhs, zb, h, K, A_visc, N, lc, mdot, beta, abs_ub, ieb, dx2, dy2, p_atm, rho_w, rho_sw, rho_i, ggrav, n, n_minus_1, kfs, b, D_x, D_y, diffusion_on)

    nx, ny = size(mask, 1), size(mask, 2)

    if ix <= nx && iy <= ny

        row = ix + (iy - 1) * nx # rhs stays flat (see MatrixFreeLinearSystem), so it still needs a linear index

        m = mask[ix, iy]

        if m == OCEAN || m == LAND # Dirichlet BC

            aP[ix, iy] = 1
            rhs[row] = dirichlet_head(m, zb[ix, iy], p_atm, rho_w, rho_sw, ggrav)

        elseif m == OTHER_BASIN || m == FROZEN_BED # Dirichlet BC

            aP[ix, iy] = 1
            rhs[row] = h[ix, iy]

        else
            # m == GROUNDED: dynamic hydrology. Same face/aP logic as update_SALS_elliptic_kernel!.

            aE_ij = (ix < nx) ? boundary_K_face(kfs, K, mask, ix, iy, ix+1, iy) / dx2 : zero(dx2)
            aW_ij = (ix > 1)  ? boundary_K_face(kfs, K, mask, ix, iy, ix-1, iy) / dx2 : zero(dx2)
            aN_ij = (iy < ny) ? boundary_K_face(kfs, K, mask, ix, iy, ix, iy+1) / dy2 : zero(dy2)
            aS_ij = (iy > 1)  ? boundary_K_face(kfs, K, mask, ix, iy, ix, iy-1) / dy2 : zero(dy2)

            aP[ix, iy] = (aE_ij + aW_ij + aN_ij + aS_ij) + n * rho_w * ggrav * A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * lc[ix, iy]

            # As in update_SALS_elliptic_kernel!: an OCEAN/LAND neighbour's known head
            # is folded into rhs instead of being wired up as a matrix
            # coupling (aE/aW/aN/aS stay at their fill!-ed 0 for that
            # direction), so the operator stays symmetric.
            dirichlet_rhs = zero(eltype(rhs))
            if ix < nx
                mE = mask[ix+1, iy]
                (mE == OCEAN || mE == LAND) ? (dirichlet_rhs += aE_ij * dirichlet_head(mE, zb[ix+1, iy], p_atm, rho_w, rho_sw, ggrav)) : (aE[ix, iy] = aE_ij)
            end
            if ix > 1
                mW = mask[ix-1, iy]
                (mW == OCEAN || mW == LAND) ? (dirichlet_rhs += aW_ij * dirichlet_head(mW, zb[ix-1, iy], p_atm, rho_w, rho_sw, ggrav)) : (aW[ix, iy] = aW_ij)
            end
            if iy < ny
                mN = mask[ix, iy+1]
                (mN == OCEAN || mN == LAND) ? (dirichlet_rhs += aN_ij * dirichlet_head(mN, zb[ix, iy+1], p_atm, rho_w, rho_sw, ggrav)) : (aN[ix, iy] = aN_ij)
            end
            if iy > 1
                mS = mask[ix, iy-1]
                (mS == OCEAN || mS == LAND) ? (dirichlet_rhs += aS_ij * dirichlet_head(mS, zb[ix, iy-1], p_atm, rho_w, rho_sw, ggrav)) : (aS[ix, iy] = aS_ij)
            end

            rhs[row] = mdot[ix, iy] * (1 / rho_w - 1 / rho_i) -
                        beta[ix, iy] * abs_ub[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * lc[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * (n * rho_w * ggrav * h[ix, iy]) * lc[ix, iy] +
                        ieb[ix, iy] +
                        diffusion_source(diffusion_on, D_x, D_y, mask, b, ix, iy, nx, ny, dx2, dy2) + # -div(D*grad(b)) (SUHMO Eq. 11), zero under NoDiffusion
                        dirichlet_rhs
        end

    end

    return
end

"""
$(TYPEDSIGNATURES)

Rebuilds `mfls.aP`/`aE`/`aW`/`aN`/`aS`/`rhs` in place from the current `s`/`p` -- the
[`MatrixFreeLinearSystem`](@ref) counterpart of [`update_SALS_elliptic!`](@ref) (same per-cell logic, no
sparse matrix to address into).
"""
function update_MFLS_elliptic!(mfls::MatrixFreeLinearSystem, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, ds::AbstractDiffusionScheme = NoDiffusion())

    fill!(mfls.aP, 0)
    fill!(mfls.aE, 0)
    fill!(mfls.aW, 0)
    fill!(mfls.aN, 0)
    fill!(mfls.aS, 0)
    fill!(mfls.rhs, 0)

    @parallel update_MFLS_elliptic_kernel!(s.mask, mfls.aP, mfls.aE, mfls.aW, mfls.aN, mfls.aS, mfls.rhs, s.zb, s.h, s.K, s.A_visc, s.N, s.lc, s.mdot, s.beta, s.abs_ub, s.ieb, g.dx2, g.dy2, p.p_atm, p.rho_w, p.rho_sw, p.rho_i, p.g, p.n, p.n_minus_1_exp, kfs, s.b, s.D_x, s.D_y, Val(ds isa WithDiffusion))

    return

end

# Sibling of update_MFLS_elliptic_kernel! for the parabolic head scheme -- the
# matrix-free counterpart of update_SALS_parabolic_kernel! (same per-cell
# logic, no sparse matrix to address into). See update_SALS_parabolic_kernel!
# for the storage-term/Newton-linearization reasoning, and for why `h_old`
# (fixed for the whole real timestep) must be a separate argument from `h`
# (the current Picard sub-iterate).
@parallel_indices (ix, iy) function update_MFLS_parabolic_kernel!(mask, aP, aE, aW, aN, aS, rhs, zb, h, h_old, K, A_visc, N, lc, mdot, beta, abs_ub, ieb, dx2, dy2, p_atm, rho_w, rho_sw, rho_i, ggrav, n, n_minus_1, kfs, e_v, dt, b, D_x, D_y, diffusion_on)

    nx, ny = size(mask, 1), size(mask, 2)

    if ix <= nx && iy <= ny

        row = ix + (iy - 1) * nx

        m = mask[ix, iy]

        if m == OCEAN || m == LAND # Dirichlet BC

            aP[ix, iy] = 1
            rhs[row] = dirichlet_head(m, zb[ix, iy], p_atm, rho_w, rho_sw, ggrav)

        elseif m == OTHER_BASIN || m == FROZEN_BED # Dirichlet BC

            aP[ix, iy] = 1
            rhs[row] = h[ix, iy]

        else
            # m == GROUNDED: dynamic hydrology. Same face logic as update_MFLS_elliptic_kernel!.

            aE_ij = (ix < nx) ? boundary_K_face(kfs, K, mask, ix, iy, ix+1, iy) / dx2 : zero(dx2)
            aW_ij = (ix > 1)  ? boundary_K_face(kfs, K, mask, ix, iy, ix-1, iy) / dx2 : zero(dx2)
            aN_ij = (iy < ny) ? boundary_K_face(kfs, K, mask, ix, iy, ix, iy+1) / dy2 : zero(dy2)
            aS_ij = (iy > 1)  ? boundary_K_face(kfs, K, mask, ix, iy, ix, iy-1) / dy2 : zero(dy2)

            aP[ix, iy] = (aE_ij + aW_ij + aN_ij + aS_ij) + e_v / dt + n * rho_w * ggrav * A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * lc[ix, iy] # diffusion + backward-Euler englacial storage reaction term + Newton-linearized creep closure (same term as update_MFLS_elliptic_kernel!'s aP)

            dirichlet_rhs = zero(eltype(rhs))
            if ix < nx
                mE = mask[ix+1, iy]
                (mE == OCEAN || mE == LAND) ? (dirichlet_rhs += aE_ij * dirichlet_head(mE, zb[ix+1, iy], p_atm, rho_w, rho_sw, ggrav)) : (aE[ix, iy] = aE_ij)
            end
            if ix > 1
                mW = mask[ix-1, iy]
                (mW == OCEAN || mW == LAND) ? (dirichlet_rhs += aW_ij * dirichlet_head(mW, zb[ix-1, iy], p_atm, rho_w, rho_sw, ggrav)) : (aW[ix, iy] = aW_ij)
            end
            if iy < ny
                mN = mask[ix, iy+1]
                (mN == OCEAN || mN == LAND) ? (dirichlet_rhs += aN_ij * dirichlet_head(mN, zb[ix, iy+1], p_atm, rho_w, rho_sw, ggrav)) : (aN[ix, iy] = aN_ij)
            end
            if iy > 1
                mS = mask[ix, iy-1]
                (mS == OCEAN || mS == LAND) ? (dirichlet_rhs += aS_ij * dirichlet_head(mS, zb[ix, iy-1], p_atm, rho_w, rho_sw, ggrav)) : (aS[ix, iy] = aS_ij)
            end

            rhs[row] = mdot[ix, iy] * (1 / rho_w - 1 / rho_i) -
                        beta[ix, iy] * abs_ub[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * lc[ix, iy] +
                        A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * (n * rho_w * ggrav * h[ix, iy]) * lc[ix, iy] + # term from Newton linearization of the creep closing term
                        ieb[ix, iy] +
                        diffusion_source(diffusion_on, D_x, D_y, mask, b, ix, iy, nx, ny, dx2, dy2) + # -div(D*grad(b)) (SUHMO Eq. 11), zero under NoDiffusion
                        (e_v / dt) * h_old[ix, iy] +
                        dirichlet_rhs
        end

    end

    return
end

"""
$(TYPEDSIGNATURES)

Rebuilds `mfls.aP`/`aE`/`aW`/`aN`/`aS`/`rhs` in place for the parabolic head scheme
([`update_MFLS_parabolic_kernel!`](@ref)) -- the [`MatrixFreeLinearSystem`](@ref) counterpart of
[`update_SALS_parabolic!`](@ref), including its `h_old` argument (see that function's docstring).
"""
function update_MFLS_parabolic!(mfls::MatrixFreeLinearSystem, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, dt, h_old, ds::AbstractDiffusionScheme = NoDiffusion())

    fill!(mfls.aP, 0)
    fill!(mfls.aE, 0)
    fill!(mfls.aW, 0)
    fill!(mfls.aN, 0)
    fill!(mfls.aS, 0)
    fill!(mfls.rhs, 0)

    @parallel update_MFLS_parabolic_kernel!(s.mask, mfls.aP, mfls.aE, mfls.aW, mfls.aN, mfls.aS, mfls.rhs, s.zb, s.h, h_old, s.K, s.A_visc, s.N, s.lc, s.mdot, s.beta, s.abs_ub, s.ieb, g.dx2, g.dy2, p.p_atm, p.rho_w, p.rho_sw, p.rho_i, p.g, p.n, p.n_minus_1_exp, kfs, p.e_v, dt, s.b, s.D_x, s.D_y, Val(ds isa WithDiffusion))

    return

end

# =============================================================================
# b's own implicit diffusion solve (SUHMO Eq. 17, WithDiffusion only)
# =============================================================================
# Unlike h's operator, b's has no real Dirichlet boundary: every GROUNDED cell
# solves the diffusion-coupled equation below, every non-GROUNDED cell gets a
# frozen identity row (b held at its current value) -- the same "not evolved
# here" convention compute_b_implicit_kernel!/etc. already use (they only
# ever touch GROUNDED cells), just expressed as a solvable row instead of a
# skipped one, since every cell needs SOME row in a system sized (nx*ny).
# Face coupling uses the exact same "neighbour must be GROUNDED" Neumann rule
# as diffusion_source above (not just the domain edge) -- see that function's
# own docstring for why OCEAN/LAND get the same treatment as OTHER_BASIN/
# FROZEN_BED here, unlike h's operator.
#
# Local (opening/closure) terms are evaluated explicitly, at the lagged b/N/
# beta/lc (matching Eq. 17: only the spatial term is implicit) -- the exact
# same RHS ImplicitGapScheme's own compute_b_implicit_kernel! would produce
# under StandardCreep, just here it becomes the right-hand side of a global
# solve instead of the final per-cell answer.
"""
$(TYPEDSIGNATURES)

Builds `b`'s own diffusion operator `(I - dt*∇·D∇)` and RHS (SUHMO Eq. 17) into `sals` -- see the
module-level note just above for the boundary convention and the local-term treatment. `aP = 1 +
dt*(D_E+D_W)/dx2 + dt*(D_N+D_S)/dy2` (the `1` is the equation's own "`I`"), each face coupling
`dt*D_face/dx2_or_dy2` if that neighbour is `GROUNDED`, else `0` -- symmetric by construction (a
shared face's `D` value is read identically from both sides, same reasoning as `h`'s own operator).
"""
@parallel_indices (ix, iy) function update_SALS_b_diffusion_kernel!(mask, nzval, rhs, idxP, idxE, idxW, idxN, idxS, b, mdot, beta, abs_ub, A_visc, N, lc, D_x, D_y, rho_i, n_minus_1, dx2, dy2, dt)

    nx, ny = size(mask, 1), size(mask, 2)

    if ix <= nx && iy <= ny

        row = ix + (iy - 1) * nx
        m = mask[ix, iy]

        if m == GROUNDED

            aE = (ix < nx && mask[ix+1, iy] == GROUNDED) ? dt * D_x[ix+1, iy] / dx2 : zero(dt)
            aW = (ix > 1  && mask[ix-1, iy] == GROUNDED) ? dt * D_x[ix, iy]   / dx2 : zero(dt)
            aN = (iy < ny && mask[ix, iy+1] == GROUNDED) ? dt * D_y[ix, iy+1] / dy2 : zero(dt)
            aS = (iy > 1  && mask[ix, iy-1] == GROUNDED) ? dt * D_y[ix, iy]   / dy2 : zero(dt)

            nzval[idxP[ix, iy]] = one(aE) + aE + aW + aN + aS
            ix < nx && (nzval[idxE[ix, iy]] = -aE)
            ix > 1  && (nzval[idxW[ix, iy]] = -aW)
            iy < ny && (nzval[idxN[ix, iy]] = -aN)
            iy > 1  && (nzval[idxS[ix, iy]] = -aS)

            rhs[row] = b[ix, iy] + dt * (mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy] -
                            A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * lc[ix, iy])

        else # not evolved: frozen at its current value, same convention as compute_b_implicit_kernel!'s own GROUNDED-only guard

            nzval[idxP[ix, iy]] = 1
            rhs[row] = b[ix, iy]

        end

    end

    return
end

"""
$(TYPEDSIGNATURES)

Rebuilds `sals.M`/`sals.rhs` in place for `b`'s own diffusion solve from the current `s`/`p`/`dt`
-- see [`update_SALS_b_diffusion_kernel!`](@ref).
"""
function update_SALS_b_diffusion!(sals::SparseAssembledLinearSystem, s::State, g::Grid, p::ModelParameters, dt)

    rhs = sals.rhs
    nzval = sals.M.nzval
    idxP, idxE, idxW, idxN, idxS = sals.idxP, sals.idxE, sals.idxW, sals.idxN, sals.idxS

    fill!(nzval, 0)
    fill!(rhs, 0)

    @parallel update_SALS_b_diffusion_kernel!(s.mask, nzval, rhs, idxP, idxE, idxW, idxN, idxS, s.b, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, s.lc, s.D_x, s.D_y, p.rho_i, p.n_minus_1_exp, g.dx2, g.dy2, dt)

    return

end

# MatrixFreeLinearSystem counterpart of update_SALS_b_diffusion_kernel! -- same per-cell logic, no
# sparse matrix to address into (see update_MFLS_elliptic_kernel!'s own note on this pairing).
@parallel_indices (ix, iy) function update_MFLS_b_diffusion_kernel!(mask, aP, aE, aW, aN, aS, rhs, b, mdot, beta, abs_ub, A_visc, N, lc, D_x, D_y, rho_i, n_minus_1, dx2, dy2, dt)

    nx, ny = size(mask, 1), size(mask, 2)

    if ix <= nx && iy <= ny

        row = ix + (iy - 1) * nx
        m = mask[ix, iy]

        if m == GROUNDED

            aE_ij = (ix < nx && mask[ix+1, iy] == GROUNDED) ? dt * D_x[ix+1, iy] / dx2 : zero(dt)
            aW_ij = (ix > 1  && mask[ix-1, iy] == GROUNDED) ? dt * D_x[ix, iy]   / dx2 : zero(dt)
            aN_ij = (iy < ny && mask[ix, iy+1] == GROUNDED) ? dt * D_y[ix, iy+1] / dy2 : zero(dt)
            aS_ij = (iy > 1  && mask[ix, iy-1] == GROUNDED) ? dt * D_y[ix, iy]   / dy2 : zero(dt)

            aP[ix, iy] = one(aE_ij) + aE_ij + aW_ij + aN_ij + aS_ij
            ix < nx && (aE[ix, iy] = aE_ij)
            ix > 1  && (aW[ix, iy] = aW_ij)
            iy < ny && (aN[ix, iy] = aN_ij)
            iy > 1  && (aS[ix, iy] = aS_ij)

            rhs[row] = b[ix, iy] + dt * (mdot[ix, iy] / rho_i + beta[ix, iy] * abs_ub[ix, iy] -
                            A_visc[ix, iy] * pow(abs(N[ix, iy]), n_minus_1) * N[ix, iy] * lc[ix, iy])

        else

            aP[ix, iy] = 1
            rhs[row] = b[ix, iy]

        end

    end

    return
end

"""
$(TYPEDSIGNATURES)

Rebuilds `mfls.aP`/`aE`/`aW`/`aN`/`aS`/`rhs` in place for `b`'s own diffusion solve -- the
[`MatrixFreeLinearSystem`](@ref) counterpart of [`update_SALS_b_diffusion!`](@ref).
"""
function update_MFLS_b_diffusion!(mfls::MatrixFreeLinearSystem, s::State, g::Grid, p::ModelParameters, dt)

    fill!(mfls.aP, 0)
    fill!(mfls.aE, 0)
    fill!(mfls.aW, 0)
    fill!(mfls.aN, 0)
    fill!(mfls.aS, 0)
    fill!(mfls.rhs, 0)

    @parallel update_MFLS_b_diffusion_kernel!(s.mask, mfls.aP, mfls.aE, mfls.aW, mfls.aN, mfls.aS, mfls.rhs, s.b, s.mdot, s.beta, s.abs_ub, s.A_visc, s.N, s.lc, s.D_x, s.D_y, p.rho_i, p.n_minus_1_exp, g.dx2, g.dy2, dt)

    return

end

# The matrix-free matvec: y = A*x computed directly from the stencil
# coefficients, with no sparse matrix ever materialized. x/y arrive as flat
# Vectors (Krylov.jl's calling convention); reshape is a zero-copy view for a
# plain Vector/Matrix, so this costs nothing over indexing a Matrix directly.
@parallel_indices (ix, iy) function stencil_matvec_kernel!(y, aP, aE, aW, aN, aS, x)

    nx, ny = size(aP, 1), size(aP, 2)

    if ix <= nx && iy <= ny

        # remember the a's are matrix column entries for a given row so for a given row of the flat vector y, we have row (of matrix) times column (which is the flat x column vector) and this is what is represented here below
        yij = aP[ix, iy] * x[ix, iy]
        ix < nx && (yij -= aE[ix, iy] * x[ix+1, iy])
        ix > 1  && (yij -= aW[ix, iy] * x[ix-1, iy])
        iy < ny && (yij -= aN[ix, iy] * x[ix, iy+1])
        iy > 1  && (yij -= aS[ix, iy] * x[ix, iy-1])
        y[ix, iy] = yij

    end

    return
end

"""
$(TYPEDSIGNATURES)

Wraps a [`MatrixFreeLinearSystem`](@ref)'s stencil coefficients as a `LinearAlgebra`-compatible
linear operator (`eltype`, `size`, `mul!` are implemented below), so Krylov.jl's `cg!` can use it
directly without ever materializing a matrix.
"""
struct StencilOperator{F <: AbstractFloat, A <: AbstractMatrix{F}}
    aP::A
    aE::A
    aW::A
    aN::A
    aS::A
    nx::Int
    ny::Int
end

"""
$(TYPEDSIGNATURES)

Builds a [`StencilOperator`](@ref) view of `mfls`'s current coefficients.
"""
StencilOperator(mfls::MatrixFreeLinearSystem) = StencilOperator(mfls.aP, mfls.aE, mfls.aW, mfls.aN, mfls.aS, size(mfls.aP, 1), size(mfls.aP, 2))

# These four functions below are required by the cg! from Krylov.jl 
Base.eltype(::StencilOperator{F}) where F = F
Base.size(op::StencilOperator) = (op.nx * op.ny, op.nx * op.ny)
Base.size(op::StencilOperator, i::Int) = size(op)[i]

function LinearAlgebra.mul!(y::AbstractVector, op::StencilOperator, x::AbstractVector)
    x2 = reshape(x, op.nx, op.ny)
    y2 = reshape(y, op.nx, op.ny)
    @parallel stencil_matvec_kernel!(y2, op.aP, op.aE, op.aW, op.aN, op.aS, x2)
    return y
end

# Jacobi (diagonal) preconditioner: refresh a preallocated diagonal vector
# from whichever linear-system representation is in use. Passed to Krylov.jl
# as `M = Diagonal(d), ldiv = true`, so Krylov calls ldiv!(y, Diagonal(d), x)
# i.e. y = x ./ d -- LinearAlgebra.Diagonal already implements this, no
# custom preconditioner type needed.
"""
$(TYPEDSIGNATURES)

Refreshes the Jacobi (diagonal) preconditioner vector `d` from `sals`'s current diagonal
(`sals.M.nzval[sals.idxP]`). Passed to Krylov.jl as `M = Diagonal(d), ldiv = true`.
"""
function update_diag_precond!(d::AbstractVector, sals::SparseAssembledLinearSystem)
    nzval = sals.M.nzval
    idxP = sals.idxP
    @inbounds for i in eachindex(idxP)
        d[i] = nzval[idxP[i]]
    end
    return d
end

"""
$(TYPEDSIGNATURES)

Refreshes the Jacobi (diagonal) preconditioner vector `d` from `mfls`'s current `aP`, the
[`MatrixFreeLinearSystem`](@ref) counterpart of the `SparseAssembledLinearSystem` method above.
"""
function update_diag_precond!(d::AbstractVector, mfls::MatrixFreeLinearSystem)
    d .= vec(mfls.aP)
    return d
end

"""
$(TYPEDSIGNATURES)

Solves the linearized elliptic equation for `h`, storing the result in `s.h` -- the
[`CholeskyDirectSolver`](@ref) method: rebuilds the system ([`update_SALS_elliptic!`](@ref)),
refactorizes in place, and solves via `\\` (see the module note on why not `ldiv!`).
"""
function solve_elliptic_linear_system!(ls::CholeskyDirectSolver, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, ds::AbstractDiffusionScheme = NoDiffusion())

    update_SALS_elliptic!(ls.sals, s, g, p, kfs, ds) # prepare the new linear system sparse matrix M and rhs

    cholesky!(ls.fact, Symmetric(ls.sals.M)) # refactorizes in-place, reusing ls.fact's symbolic factorization since the sparsity pattern never changes across Picard iterations/timesteps

    # NOT ldiv!(ls.h_vec, ls.fact, ls.sals.rhs): CHOLMOD's in-place ldiv! leaks native (non-GC-tracked)
    # memory on every call, regardless of whether the output buffer is reused or fresh -- confirmed
    # directly (JuliaSparse/SparseArrays.jl#726, unfixed as of Julia 1.12.7: RSS climbs ~56KB/call,
    # unrecoverable via GC.gc(), identical rate whether ls.h_vec is reused or a fresh vector is
    # passed each time). `\` avoids the leaky path entirely (confirmed flat RSS over 20k calls), at
    # the cost of one small ordinary (GC'd) Vector allocation per solve instead of an unbounded leak.
    ls.h_vec .= ls.fact \ ls.sals.rhs

    s.h .= reshape(ls.h_vec, g.nx, g.ny) # update h

end

"""
$(TYPEDSIGNATURES)

Solves the backward-Euler parabolic equation for `h` (`p.e_v != 0`), storing the result in `s.h`
-- the [`CholeskyDirectSolver`](@ref) method: rebuilds the system
([`update_SALS_parabolic!`](@ref)), refactorizes in place, and solves via `\\` (see
[`solve_elliptic_linear_system!`](@ref)'s note on why not `ldiv!`). `h_old` is the head at the
*start of the real timestep* (fixed across every [`Parabolic_loop!`](@ref) sub-iteration, unlike
`s.h` itself) -- see [`update_SALS_parabolic!`](@ref)'s docstring for why it must be threaded
through explicitly rather than read off `s.h`.
"""
function solve_parabolic_linear_system!(ls::CholeskyDirectSolver, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, dt, h_old, ds::AbstractDiffusionScheme = NoDiffusion())

    update_SALS_parabolic!(ls.sals, s, g, p, kfs, dt, h_old, ds)

    cholesky!(ls.fact, Symmetric(ls.sals.M))

    ls.h_vec .= ls.fact \ ls.sals.rhs

    s.h .= reshape(ls.h_vec, g.nx, g.ny)

end

"""
$(TYPEDSIGNATURES)

Solves `b`'s own coupled diffusion system (SUHMO Eq. 17) via `ls`, storing the (clamped to
`p.b_min`/`p.b_max`, same convention every other `compute_b!` method uses) result in `s.b` -- the
[`CholeskyDirectSolver`](@ref) method. `ls` here is the *second*, independent solver instance
bundled in [`WithDiffusion`](@ref) (built the same way as `h`'s own, e.g. a second
`CholeskyDirectSolver(grid)`) -- reuses `ls.h_vec` as a generic solve-result scratch buffer despite
its name; this instance never actually solves for `h`.
"""
function solve_b_diffusion!(ls::CholeskyDirectSolver, s::State, g::Grid, p::ModelParameters, dt)

    update_SALS_b_diffusion!(ls.sals, s, g, p, dt)

    cholesky!(ls.fact, Symmetric(ls.sals.M))

    ls.h_vec .= ls.fact \ ls.sals.rhs

    s.b .= clamp.(reshape(ls.h_vec, g.nx, g.ny), p.b_min, p.b_max)

end

"""
$(TYPEDSIGNATURES)

Iterative solve via preconditioned conjugate gradient (Krylov.jl's `cg!`) -- valid because the
assembled operator is symmetric positive definite (diffusion with reciprocal face fluxes plus a
strictly positive diagonal reaction term from the Newton-linearized creep closure, Dirichlet
neighbours eliminated symmetrically). Works over either linear-system representation
([`SparseAssembledLinearSystem`](@ref) or [`MatrixFreeLinearSystem`](@ref), chosen via the `LSy`
type parameter at construction, see the constructors below) and warm-starts from the previous
head each solve.

# Notes

`precond` picks the preconditioner: `nothing` for plain Jacobi, or an `AMGPreconditioner`/
`ChebyshevPreconditioner` (see `preconditioner.jl`) for a much faster-converging accelerated
preconditioner -- AMG is the default for `SparseAssembledLinearSystem` (near mesh-independent CG
iteration counts; measured to consistently beat both plain Jacobi and Chebyshev) but is CPU/`SparseMatrixCSC`-only, 
so `MatrixFreeLinearSystem` defaults to plain Jacobi with Chebyshev available as a GPU-capable accelerated option instead.
"""
struct CGIterativeSolver{LSy <: AbstractLinearSystem, WS, V <: AbstractVector, P} <: AbstractIterativeSolver
    lsy::LSy # linear system can be SALS or MFLS (sparse assembled matrix or matrix free linear system)
    ws::WS # workspace
    precond_diag::V # Jacobi (diagonal) preconditioner, refreshed every solve -- same array type as lsy's rhs (Vector for SALS, backend-native for MatrixFreeLinearSystem)
    precond::P # `nothing` -> plain Jacobi (Diagonal(precond_diag)); ChebyshevPreconditioner/AMGPreconditioner -> see preconditioner.jl
end

# Representation is chosen at construction time by passing the AbstractLinearSystem
# subtype itself, e.g. CGIterativeSolver(g, SparseAssembledLinearSystem) or
# CGIterativeSolver(g, MatrixFreeLinearSystem) -- same struct either way,
# dispatch on LSy in solve_elliptic_linear_system! picks the right solve path.
#
# CG is valid here because the assembled operator
# is symmetric positive definite: diffusion with reciprocal face fluxes (see
# compute_K_face) plus a strictly positive diagonal reaction term from the
# Newton-linearized creep closure, and Dirichlet (OCEAN/LAND) neighbours are
# eliminated symmetrically (folded into rhs, see update_SALS_elliptic_kernel!/
# update_MFLS_elliptic_kernel!) rather than left as a one-sided matrix coupling.
#
# amg = true (default) opts into AMGPreconditioner -- near mesh-independent
# CG iteration counts, at the cost of a hierarchy-rebuild every solve; see
# preconditioner.jl. Measured to consistently beat both plain Jacobi and ChebyshevPreconditioner, from a
# near-initial-condition state (5-7 CG iterations vs Jacobi's 192-745) all
# the way through peak seasonal channelization (24 iterations vs Jacobi's
# 199 and Chebyshev's 71, at K max/min ratio ~14000) -- hence the default.
# Passing chebyshev_degree an Int opts into ChebyshevPreconditioner instead
# -- see preconditioner.jl for why (GPU host-sync overhead) and how (Saad's
# Chebyshev semi-iteration, layered on top of the same Jacobi scaling); amg
# and chebyshev_degree are mutually exclusive (both are full replacements
# for Jacobi, not composable with each other). amg = false and
# chebyshev_degree = nothing together give plain Jacobi. amg is SALS-only
# (AlgebraicMultigrid.jl has no GPU array support), hence defaulting to
# false and not offered at all on the MatrixFreeLinearSystem constructor below.
"""
$(TYPEDSIGNATURES)

Builds a [`CGIterativeSolver`](@ref) over a [`SparseAssembledLinearSystem`](@ref) on grid `g`.
`amg = true` (default) opts into `AMGPreconditioner`; `chebyshev_degree` (an `Int`) opts into
`ChebyshevPreconditioner` instead (mutually exclusive with `amg`); both `false`/`nothing` gives
plain Jacobi. CPU-only, same reasoning as [`CholeskyDirectSolver`](@ref).
"""
function CGIterativeSolver(g::Grid{F}, ::Type{SparseAssembledLinearSystem}; chebyshev_degree::Union{Nothing, Int} = nothing, chebyshev_nsteps_estimate::Int = 15, amg::Bool = true) where F

    # Krylov.jl's sparse matvec (SparseArrays.mul!) is CPU-only, same reasoning as CholeskyDirectSolver.
    backend != "Threads" && error("CGIterativeSolver(g, SparseAssembledLinearSystem) is CPU-only; use CGIterativeSolver(g, MatrixFreeLinearSystem) under the $backend backend.")

    (chebyshev_degree !== nothing && amg) && error("chebyshev_degree and amg are mutually exclusive preconditioner choices for CGIterativeSolver; pick one.")

    sals = SparseAssembledLinearSystem(g)
    ws = CgWorkspace(sals.M, sals.rhs)
    precond_diag = zeros(F, g.nx * g.ny)
    precond = if amg
        AMGPreconditioner(sals.M)
    elseif chebyshev_degree !== nothing
        ChebyshevPreconditioner(sals.M, precond_diag, chebyshev_degree; nsteps_estimate = chebyshev_nsteps_estimate)
    else
        nothing
    end

    return CGIterativeSolver(sals, ws, precond_diag, precond)

end

"""
$(TYPEDSIGNATURES)

Builds a [`CGIterativeSolver`](@ref) over a [`MatrixFreeLinearSystem`](@ref) on grid `g`
(GPU-capable). `chebyshev_degree` (an `Int`) opts into `ChebyshevPreconditioner`; `nothing`
(default) gives plain Jacobi. `amg = true` errors -- `AMGPreconditioner` needs a
`SparseMatrixCSC`, use the [`CGIterativeSolver`](@ref) constructor over
[`SparseAssembledLinearSystem`](@ref) instead.
"""
function CGIterativeSolver(g::Grid{F}, ::Type{MatrixFreeLinearSystem}; chebyshev_degree::Union{Nothing, Int} = nothing, chebyshev_nsteps_estimate::Int = 15, amg::Bool = false) where F

    amg && error("AMGPreconditioner is CPU/SparseMatrixCSC-only (AlgebraicMultigrid.jl has no GPU array support); use CGIterativeSolver(g, SparseAssembledLinearSystem; amg = true) instead, or chebyshev_degree here for a GPU-capable accelerated preconditioner.")

    mfls = MatrixFreeLinearSystem(g)
    ws = CgWorkspace(g.nx * g.ny, g.nx * g.ny, typeof(mfls.rhs)) # storage type matches mfls.rhs, so it lands on the active backend (Array under Threads, MtlArray under Metal)
    precond_diag = @zeros(g.nx * g.ny)
    precond = chebyshev_degree === nothing ? nothing : ChebyshevPreconditioner(StencilOperator(mfls), precond_diag, chebyshev_degree; nsteps_estimate = chebyshev_nsteps_estimate)

    return CGIterativeSolver(mfls, ws, precond_diag, precond)

end

# One method per precond choice, made once at construction time (ls.precond's
# type), not re-decided per solve. Each `where` clause has to restate
# CGIterativeSolver's own parameter bounds (LSy <: AbstractLinearSystem,
# V <: AbstractVector) -- omitting them (or using bare `<:Any`) makes these
# ambiguous with each other instead of each strictly most-specific for its
# own P, and Julia silently picks the wrong one rather than erroring.
_cg_precond!(ls::CGIterativeSolver{LSy, WS, V, Nothing}) where {LSy <: AbstractLinearSystem, WS, V <: AbstractVector} = Diagonal(ls.precond_diag)

function _cg_precond!(ls::CGIterativeSolver{LSy, WS, V, <:ChebyshevPreconditioner}) where {LSy <: AbstractLinearSystem, WS, V <: AbstractVector}
    update_chebyshev_bounds!(ls.precond, ls.lsy.rhs) # cheap (nsteps_estimate matvecs), but real -- see preconditioner.jl for why this can't just be done once at construction
    return ls.precond
end

function _cg_precond!(ls::CGIterativeSolver{<:SparseAssembledLinearSystem, WS, V, <:AMGPreconditioner}) where {WS, V <: AbstractVector}
    update_amg!(ls.precond, ls.lsy.M) # hierarchy rebuild, not free -- see preconditioner.jl for why this can't just be done once at construction
    return ls.precond
end

"""
$(TYPEDSIGNATURES)

Solves the linearized elliptic equation for `h`, storing the result in `s.h` -- the
[`CGIterativeSolver`](@ref) over [`SparseAssembledLinearSystem`](@ref) method: rebuilds the
system, refreshes the preconditioner, and runs `cg!` warm-started from the current `s.h`.
"""
function solve_elliptic_linear_system!(ls::CGIterativeSolver{<:SparseAssembledLinearSystem}, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, ds::AbstractDiffusionScheme = NoDiffusion())

    update_SALS_elliptic!(ls.lsy, s, g, p, kfs, ds) # prepare the new linear system sparse matrix M and rhs
    update_diag_precond!(ls.precond_diag, ls.lsy)
    precond_matrix = _cg_precond!(ls)

    # vec(s.h) as x0 warm-starts from the previous head instead of 0: cheap (Krylov just
    # copies it into its own Δx buffer) and correctness-neutral (Krylov converges to the
    # same solution regardless of x0), but the residual it starts from is usually much
    # smaller once h is already close to converged (late Picard iterations, or consecutive
    # time steps), so it typically needs fewer Krylov iterations.
    cg!(ls.ws, ls.lsy.M, ls.lsy.rhs, vec(s.h); M = precond_matrix, ldiv = true) # solves in place, storing the result in the preallocated workspace ls.ws

    s.h .= reshape(ls.ws.x, g.nx, g.ny) # update h

end

"""
$(TYPEDSIGNATURES)

Solves the backward-Euler parabolic equation for `h` (`p.e_v != 0`), storing the result in `s.h`
-- the [`CGIterativeSolver`](@ref) over [`SparseAssembledLinearSystem`](@ref) method: rebuilds the
system ([`update_SALS_parabolic!`](@ref)), refreshes the preconditioner, and runs `cg!` warm-started
from the current `s.h`. `h_old` is the head at the *start of the real timestep* -- see
[`update_SALS_parabolic!`](@ref)'s docstring.
"""
function solve_parabolic_linear_system!(ls::CGIterativeSolver{<:SparseAssembledLinearSystem}, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, dt, h_old, ds::AbstractDiffusionScheme = NoDiffusion())

    update_SALS_parabolic!(ls.lsy, s, g, p, kfs, dt, h_old, ds)
    update_diag_precond!(ls.precond_diag, ls.lsy)
    precond_matrix = _cg_precond!(ls)

    cg!(ls.ws, ls.lsy.M, ls.lsy.rhs, vec(s.h); M = precond_matrix, ldiv = true)

    s.h .= reshape(ls.ws.x, g.nx, g.ny)

end

"""
$(TYPEDSIGNATURES)

Solves `b`'s own coupled diffusion system (SUHMO Eq. 17), storing the (clamped) result in `s.b` --
the [`CGIterativeSolver`](@ref) over [`SparseAssembledLinearSystem`](@ref) method, warm-started
from the current `s.b` (a good starting guess: `b`'s own value from the previous timestep).
"""
function solve_b_diffusion!(ls::CGIterativeSolver{<:SparseAssembledLinearSystem}, s::State, g::Grid, p::ModelParameters, dt)

    update_SALS_b_diffusion!(ls.lsy, s, g, p, dt)
    update_diag_precond!(ls.precond_diag, ls.lsy)
    precond_matrix = _cg_precond!(ls)

    cg!(ls.ws, ls.lsy.M, ls.lsy.rhs, vec(s.b); M = precond_matrix, ldiv = true)

    s.b .= clamp.(reshape(ls.ws.x, g.nx, g.ny), p.b_min, p.b_max)

end

"""
$(TYPEDSIGNATURES)

Solves the linearized elliptic equation for `h`, storing the result in `s.h` -- the
[`CGIterativeSolver`](@ref) over [`MatrixFreeLinearSystem`](@ref) method, using
[`StencilOperator`](@ref) as `cg!`'s matvec.
"""
function solve_elliptic_linear_system!(ls::CGIterativeSolver{<:MatrixFreeLinearSystem}, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, ds::AbstractDiffusionScheme = NoDiffusion())

    update_MFLS_elliptic!(ls.lsy, s, g, p, kfs, ds)
    update_diag_precond!(ls.precond_diag, ls.lsy)
    precond_matrix = _cg_precond!(ls)

    cg!(ls.ws, StencilOperator(ls.lsy), ls.lsy.rhs, vec(s.h); M = precond_matrix, ldiv = true)

    s.h .= reshape(ls.ws.x, g.nx, g.ny) # update h

end

"""
$(TYPEDSIGNATURES)

Solves the backward-Euler parabolic equation for `h` (`p.e_v != 0`), storing the result in `s.h`
-- the [`CGIterativeSolver`](@ref) over [`MatrixFreeLinearSystem`](@ref) method, using
[`StencilOperator`](@ref) as `cg!`'s matvec. `h_old` is the head at the *start of the real
timestep* -- see [`update_SALS_parabolic!`](@ref)'s docstring.
"""
function solve_parabolic_linear_system!(ls::CGIterativeSolver{<:MatrixFreeLinearSystem}, s::State, g::Grid, p::ModelParameters, kfs::AbstractKFaceScheme, dt, h_old, ds::AbstractDiffusionScheme = NoDiffusion())

    update_MFLS_parabolic!(ls.lsy, s, g, p, kfs, dt, h_old, ds)
    update_diag_precond!(ls.precond_diag, ls.lsy)
    precond_matrix = _cg_precond!(ls)

    cg!(ls.ws, StencilOperator(ls.lsy), ls.lsy.rhs, vec(s.h); M = precond_matrix, ldiv = true)

    s.h .= reshape(ls.ws.x, g.nx, g.ny)

end

"""
$(TYPEDSIGNATURES)

Solves `b`'s own coupled diffusion system (SUHMO Eq. 17), storing the (clamped) result in `s.b` --
the [`CGIterativeSolver`](@ref) over [`MatrixFreeLinearSystem`](@ref) method, using
[`StencilOperator`](@ref) as `cg!`'s matvec, warm-started from the current `s.b`.
"""
function solve_b_diffusion!(ls::CGIterativeSolver{<:MatrixFreeLinearSystem}, s::State, g::Grid, p::ModelParameters, dt)

    update_MFLS_b_diffusion!(ls.lsy, s, g, p, dt)
    update_diag_precond!(ls.precond_diag, ls.lsy)
    precond_matrix = _cg_precond!(ls)

    cg!(ls.ws, StencilOperator(ls.lsy), ls.lsy.rhs, vec(s.b); M = precond_matrix, ldiv = true)

    s.b .= clamp.(reshape(ls.ws.x, g.nx, g.ny), p.b_min, p.b_max)

end
