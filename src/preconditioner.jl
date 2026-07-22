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
    P.lambda_min, P.lambda_max = estimate_eigenvalue_bounds(P.op, rhs, P.nsteps_estimate)
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
