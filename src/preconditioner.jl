# =============================================================================
# Chebyshev-polynomial preconditioning for CGIterativeSolver
# =============================================================================
#
# Jacobi/diagonal preconditioning alone leaves CG's iteration count scaling
# like O(sqrt(N)) for this 2D elliptic problem, and every CG iteration needs
# 2-3 dot products whose scalar result has to reach the host before Krylov.jl
# can decide whether to keep iterating -- on GPU that's a real GPU->CPU sync
# per iteration (see the CUDA.@profile findings that drove this file: at
# 64x64, device-side GPU compute was only ~9% of wall time, the rest was
# host-side overhead from exactly these per-iteration syncs).
#
# Chebyshev semi-iteration sidesteps this: it approximates A^-1 by a
# low-degree polynomial in the (Jacobi-scaled) operator, applied via a fixed
# three-term recurrence (Saad, "Iterative Methods for Sparse Linear Systems",
# 2nd ed., Algorithm 12.1) that uses ONLY matvecs and axpy-style updates --
# no dot products, no host syncs, at all. It needs the operator's extreme
# eigenvalues to build that polynomial, which is the one place this still
# needs reductions -- but that only has to happen once per solve (see
# update_chebyshev_bounds!), not once per CG iteration.

# Applies the diagonal-Jacobi-scaled operator D^-1*A. The same wrapper works
# for both SparseAssembledLinearSystem's SparseMatrixCSC and
# MatrixFreeLinearSystem's StencilOperator, since both already have a mul!
# method -- nothing here is backend- or representation-specific.
struct JacobiScaledOperator{Op, V <: AbstractVector}
    A::Op
    d::V # Jacobi diagonal -- a *reference* to CGIterativeSolver's precond_diag, refreshed in place by update_diag_precond! before every solve
end

function LinearAlgebra.mul!(y::AbstractVector, op::JacobiScaledOperator, p::AbstractVector)
    mul!(y, op.A, p)
    y ./= op.d
    return y
end

# Estimates [lambda_min, lambda_max] of the Jacobi-scaled operator D^-1*A via
# `nsteps` iterations of ordinary (unpreconditioned, since op already folds
# in the Jacobi scaling) CG on op, extracting the associated Lanczos
# tridiagonal matrix from the CG alpha/beta recurrence (Saad, section 6.7;
# same trick PETSc's KSPCHEBYSHEV uses to get its bounds) rather than running
# separate power-iteration machinery. A tiny (nsteps x nsteps) dense
# eigenproblem at the end is solved on the host regardless of backend --
# nsteps is a handful, so this is negligible next to the matvecs.
function estimate_eigenvalue_bounds(op::JacobiScaledOperator, rhs::AbstractVector, nsteps::Int)

    T = eltype(rhs)

    r  = copy(rhs) # r_0 = rhs - op*0 = rhs (x0 = 0)
    p  = copy(r)
    Ap = similar(r)

    gamma = dot(r, r)

    alphas = zeros(T, nsteps)
    betas  = zeros(T, nsteps - 1)

    for k in 1:nsteps
        mul!(Ap, op, p)
        pAp = dot(p, Ap)
        alphas[k] = gamma / pAp
        if k < nsteps
            r .-= alphas[k] .* Ap
            gamma_next = dot(r, r)
            betas[k] = gamma_next / gamma
            p .= r .+ betas[k] .* p
            gamma = gamma_next
        end
    end

    d = zeros(T, nsteps)
    e = zeros(T, nsteps - 1)
    d[1] = 1 / alphas[1]
    for k in 2:nsteps
        d[k] = 1 / alphas[k] + betas[k-1] / alphas[k-1]
    end
    for k in 1:nsteps-1
        e[k] = sqrt(betas[k]) / alphas[k]
    end

    ritz = eigvals(SymTridiagonal(d, e))

    # Small safety margin: the extreme Ritz values of a short Lanczos run
    # (especially the smallest) tend to still be inside the true spectrum's
    # extremes rather than exactly at them.
    lambda_min = minimum(ritz) * T(0.9)
    lambda_max = maximum(ritz) * T(1.1)

    return lambda_min, lambda_max
end

# `degree` matvecs per preconditioner application; workspace vectors
# preallocated once (same lifetime as CGIterativeSolver) and reused every
# ldiv! call. lambda_min/lambda_max are refreshed once per *solve* by
# update_chebyshev_bounds! (not on every ldiv! call within that solve's CG
# run), since the operator only changes between solves, not within one.
mutable struct ChebyshevPreconditioner{Op, V <: AbstractVector, T <: AbstractFloat}
    op::JacobiScaledOperator{Op, V}
    degree::Int
    nsteps_estimate::Int
    lambda_min::T
    lambda_max::T
    r::V
    p::V
    Ap::V
end

function ChebyshevPreconditioner(A, d::V, degree::Int; nsteps_estimate::Int = 15) where V <: AbstractVector
    T = eltype(d)
    degree >= 1 || error("ChebyshevPreconditioner: degree must be >= 1 (got $degree)")
    nsteps_estimate >= 2 || error("ChebyshevPreconditioner: nsteps_estimate must be >= 2 (got $nsteps_estimate)")
    op = JacobiScaledOperator(A, d)
    r, p, Ap = similar(d), similar(d), similar(d)
    return ChebyshevPreconditioner(op, degree, nsteps_estimate, one(T), one(T), r, p, Ap) # placeholder bounds, overwritten before first use
end

function update_chebyshev_bounds!(P::ChebyshevPreconditioner, rhs::AbstractVector)

    # The short, unreorthogonalized Lanczos recurrence in estimate_eigenvalue_bounds
    # is only exact in infinite precision -- on more ill-conditioned problems
    # (empirically, larger grids: fine at 64x64/256x256, but nonsensical/even
    # negative bounds at 512x512, non-monotonically in nsteps_estimate too,
    # the signature of numerical breakdown rather than "just needs more
    # steps" -- see test/cheb_512_diag.jl) it can lose numerical stability
    # badly enough to fail two different ways: either hand back a bogus but
    # finite interval, or -- degenerate enough input, e.g. an all-zero rhs --
    # have the tridiagonal eigensolve itself throw a LAPACKException. Either
    # way, letting that reach the Chebyshev recurrence below doesn't fail
    # loudly where the problem is -- the finite-garbage case surfaces many
    # steps downstream as Krylov.jl's cryptic "operator or preconditioner is
    # not SPD" error, and the LAPACKException case would just crash the run.
    # Falling back to the last known-valid bounds instead is always safe: on
    # the very first solve (no prior valid bounds yet) P.lambda_min/lambda_max
    # are still their construction-time placeholder (1, 1), which makes
    # ldiv!'s c_rad exactly 0 and the whole Chebyshev recurrence degenerate to
    # plain Jacobi -- a correct, if unaccelerated, fallback. On a later solve,
    # reusing the previous (valid) solve's bounds is a good approximation
    # since consecutive Picard iterations' matrices are close to each other.
    lambda_min, lambda_max = try
        estimate_eigenvalue_bounds(P.op, rhs, P.nsteps_estimate)
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        (NaN, NaN) # sentinel: fails the isfinite check below, taking the same fallback path as a bogus-but-finite estimate
    end

    if isfinite(lambda_min) && isfinite(lambda_max) && lambda_min > 0 && lambda_max > lambda_min
        P.lambda_min, P.lambda_max = lambda_min, lambda_max
    else
        @warn "ChebyshevPreconditioner: eigenvalue bound estimate was invalid (lambda_min=$lambda_min, lambda_max=$lambda_max); reusing previous bounds for this solve" maxlog = 10
    end

    return P
end

# Chebyshev semi-iteration (Saad Algorithm 12.1) approximating
# y ≈ (D^-1 A)^-1 (D^-1 x) = A^-1 x, i.e. exactly what ldiv!(y, ::AnyPreconditioner, x)
# is supposed to compute. No dot products anywhere in this loop -- only
# mul! (a matvec) and broadcasted axpy-style updates.
function LinearAlgebra.ldiv!(y::AbstractVector, P::ChebyshevPreconditioner, x::AbstractVector)

    d_mid = (P.lambda_max + P.lambda_min) / 2
    c_rad = (P.lambda_max - P.lambda_min) / 2

    r, p, Ap = P.r, P.p, P.Ap

    r .= x ./ P.op.d # r_0 = (D^-1 x) - (D^-1 A)*y_0 with y_0 = 0
    fill!(y, zero(eltype(y)))

    alpha = one(eltype(y)) / d_mid
    p .= r # p_0 = z_0 = r_0 (no secondary preconditioner here, so z_k = r_k)
    y .+= alpha .* p
    mul!(Ap, P.op, p)
    r .-= alpha .* Ap

    alpha_prev = alpha
    for _ in 2:P.degree
        beta = (c_rad * alpha_prev / 2)^2
        alpha = one(eltype(y)) / (d_mid - beta / alpha_prev)
        p .= r .+ beta .* p
        y .+= alpha .* p
        mul!(Ap, P.op, p)
        r .-= alpha .* Ap
        alpha_prev = alpha
    end

    return y
end

# =============================================================================
# Algebraic multigrid (Ruge-Stuben, AlgebraicMultigrid.jl) preconditioning for
# CGIterativeSolver{<:SparseAssembledLinearSystem}
# =============================================================================
#
# Unlike Jacobi/Chebyshev, whose CG iteration counts grow with grid size
# (Jacobi ~O(sqrt(N)), Chebyshev slower-growing but still not flat), AMG
# gives near mesh-independent convergence -- measured (test/amg_rerun.jl) at
# 5 CG iterations at 64x64 and 6 at 256x256, essentially flat despite 16x
# more unknowns, vs Jacobi's 192->745 and Chebyshev's 58->350 over the same
# grids. That makes AMG substantially faster in wall time once the grid is
# large enough for its (nontrivial, paid every solve -- see below) hierarchy
# setup cost to be worth it: ~4x faster than Jacobi and ~7.7x faster than
# Chebyshev at 256x256 in that same measurement.
#
# CPU/SparseMatrixCSC-only (AlgebraicMultigrid.jl has no GPU array support),
# so -- like CholeskyDirectSolver -- this is only offered for
# CGIterativeSolver{<:SparseAssembledLinearSystem}, not MatrixFreeLinearSystem.

mutable struct AMGPreconditioner{P}
    precond::P # AlgebraicMultigrid.jl's aspreconditioner(::MultiLevel) wrapper; already implements LinearAlgebra.ldiv!, so ldiv! below just delegates
end

# M's sparsity pattern is fixed, but its VALUES change every Picard
# iteration -- like ChebyshevPreconditioner's bounds, the AMG hierarchy
# (strength-of-connection, aggregation, interpolation) is built from those
# values, so it has to be rebuilt every solve, not just once at construction.
AMGPreconditioner(M::SparseMatrixCSC) = AMGPreconditioner(aspreconditioner(ruge_stuben(M)))

function update_amg!(P::AMGPreconditioner, M::SparseMatrixCSC)
    P.precond = aspreconditioner(ruge_stuben(M))
    return P
end

LinearAlgebra.ldiv!(y::AbstractVector, P::AMGPreconditioner, x::AbstractVector) = ldiv!(y, P.precond, x)
