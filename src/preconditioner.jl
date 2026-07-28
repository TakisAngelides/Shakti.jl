# =============================================================================
# Chebyshev-polynomial preconditioning for CGIterativeSolver
# =============================================================================
#
# Jacobi/diagonal preconditioning alone leaves CG's iteration count scaling
# like O(sqrt(N)) for this 2D elliptic problem, and every CG iteration needs
# 2-3 dot products whose scalar result has to reach the host before Krylov.jl
# can decide whether to keep iterating -- on GPU that's a real GPU->CPU sync
# per iteration.
#
# Chebyshev semi-iteration's OWN internal recurrence avoids this: it
# approximates A^-1 by a low-degree polynomial in the (Jacobi-scaled)
# operator, applied via a fixed three-term recurrence (Saad, "Iterative
# Methods for Sparse Linear Systems", 2nd ed., Algorithm 12.1) that uses ONLY
# matvecs and axpy-style updates -- no dot products, no host syncs, at all,
# within that one preconditioner-application step. It needs the operator's
# extreme eigenvalues to build that polynomial, which is the one place this
# still needs reductions -- but that only has to happen once per solve (see
# update_chebyshev_bounds!), not once per CG iteration.
#
# Caveat: using Chebyshev as CG's preconditioner does not eliminate
# CG's own per-iteration host syncs. Standard preconditioned CG still
# computes its own alpha/beta from dot products every outer iteration
# regardless of which preconditioner solves z = M^-1 r -- that's inherent to
# CG's algorithm, not something the preconditioner choice touches. The
# "no dot products" property described above is only about Chebyshev's own
# internal ldiv! recurrence in isolation, not the overall CG+Chebyshev
# iteration. Measured result: on MatrixFreeLinearSystem (the only
# GPU-capable solver) at 64x64/128x128 on an A100, plain Jacobi beat
# Chebyshev(degree=4) in wall time at both sizes (12%/29% faster
# respectively) -- Chebyshev pays the same per-iteration sync cost as Jacobi
# (from CG's own dot products) plus extra matvec work for the degree-4
# polynomial application, without cutting the number of syncs enough to
# compensate. Genuine sync-avoidance on GPU would need replacing CG itself
# with a dot-product-free outer solver (true Chebyshev semi-iteration AS the
# solver, not just the preconditioner). Per-CG-iteration cost was
# also found to be roughly flat with grid size (~229us at 64x64, ~191us at
# 128x128), consistent with fixed per-iteration overhead (kernel launch +
# host sync) dominating over device compute at these sizes on this hardware
# -- but this could not be confirmed via a proper device-vs-host profile.
#
# Net effect: for CGIterativeSolver{<:SparseAssembledLinearSystem} (CPU-only),
# AMGPreconditioner (below) is the better choice and is the default there. For
# MatrixFreeLinearSystem (the GPU-capable path, where AMGPreconditioner isn't
# available -- AlgebraicMultigrid.jl has no GPU array support), plain Jacobi
# (chebyshev_degree = nothing, the default there) should be preferred over
# Chebyshev per the above, at least at the grid sizes tested.
#
# Applies the diagonal-Jacobi-scaled operator D^-1*A. The same wrapper works
# for both SparseAssembledLinearSystem's SparseMatrixCSC and
# MatrixFreeLinearSystem's StencilOperator, since both already have a mul!
# method -- nothing here is backend- or representation-specific.
"""
$(TYPEDSIGNATURES)

Applies the diagonal-Jacobi-scaled operator `D^-1*A` via `mul!` below. The same wrapper works for
both `SparseAssembledLinearSystem`'s `SparseMatrixCSC` and `MatrixFreeLinearSystem`'s
[`StencilOperator`](@ref), since both already have a `mul!` method -- nothing here is backend- or
representation-specific. Used internally by [`ChebyshevPreconditioner`](@ref).
"""
struct JacobiScaledOperator{Op, V <: AbstractVector}
    A::Op
    d::V # Jacobi diagonal -- a *reference* to CGIterativeSolver's precond_diag, refreshed in place by update_diag_precond! before every solve
end

function LinearAlgebra.mul!(y::AbstractVector, op::JacobiScaledOperator, p::AbstractVector)
    mul!(y, op.A, p)
    y ./= op.d # Combined with the previous line this gives y = D^{-1} * A * p, i.e., left-multiplication by the inverse Jacobi diagonal.
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
"""
$(TYPEDSIGNATURES)

Estimates `[lambda_min, lambda_max]` of the Jacobi-scaled operator `op` (`D^-1*A`) via `nsteps`
iterations of ordinary (unpreconditioned) CG, extracting the associated Lanczos tridiagonal
matrix from the CG alpha/beta recurrence (Saad, section 6.7 -- the same trick PETSc's
`KSPCHEBYSHEV` uses) rather than running separate power-iteration machinery. Used by
[`update_chebyshev_bounds!`](@ref) to size [`ChebyshevPreconditioner`](@ref)'s polynomial.
"""
function estimate_eigenvalue_bounds(op::JacobiScaledOperator, rhs::AbstractVector, nsteps::Int)

    T = eltype(rhs) # scalar type to match rhs (Float32/Float64), so all work below stays in that precision

    r  = copy(rhs) # r_0 = rhs - op*0 = rhs (x0 = 0)
    p  = copy(r)   # p_0 = r_0, the initial CG search direction
    Ap = similar(r) # preallocated scratch for op*p each step

    gamma = dot(r, r) # gamma_0 = r_0'r_0, CG's running residual-norm-squared

    alphas = zeros(T, nsteps)     # CG step sizes alpha_k, one per iteration
    betas  = zeros(T, nsteps - 1) # CG direction-update coefficients beta_k, one fewer than alpha since the last iteration never updates p

    for k in 1:nsteps
        mul!(Ap, op, p)       # Ap = op*p = D^-1*A*p, the one matvec this iteration needs
        pAp = dot(p, Ap)      # p'*op*p, the denominator of the CG step size
        alphas[k] = gamma / pAp # standard CG step size alpha_k = (r_k'r_k) / (p_k'*op*p_k)
        if k < nsteps # skip the trailing residual/direction update on the last step -- alphas[nsteps] is all that's needed from it
            r .-= alphas[k] .* Ap        # update residual: r_{k+1} = r_k - alpha_k*op*p_k
            gamma_next = dot(r, r)       # gamma_{k+1} = r_{k+1}'r_{k+1}
            betas[k] = gamma_next / gamma # CG direction coefficient beta_k = gamma_{k+1}/gamma_k
            p .= r .+ betas[k] .* p      # new search direction: p_{k+1} = r_{k+1} + beta_k*p_k
            gamma = gamma_next           # carry residual norm forward to next iteration
        end
    end

    d = zeros(T, nsteps)     # diagonal of the Lanczos tridiagonal matrix built from the CG recurrence
    e = zeros(T, nsteps - 1) # off-diagonal (sub/super-diagonal, since it's symmetric) of that same matrix
    d[1] = 1 / alphas[1] # Saad eq. 6.7.15: T's first diagonal entry is 1/alpha_1
    for k in 2:nsteps
        d[k] = 1 / alphas[k] + betas[k-1] / alphas[k-1] # remaining diagonal entries combine the current and previous CG coefficients
    end
    for k in 1:nsteps-1
        e[k] = sqrt(betas[k]) / alphas[k] # off-diagonal entries from sqrt(beta_k)/alpha_k
    end

    ritz = eigvals(SymTridiagonal(d, e)) # small (nsteps x nsteps) dense eigenproblem on the host: its eigenvalues (Ritz values) approximate op's extreme eigenvalues

    # Small safety margin: the extreme Ritz values of a short Lanczos run
    # (especially the smallest) tend to still be inside the true spectrum's
    # extremes rather than exactly at them.
    lambda_min = minimum(ritz) * T(0.9) # shrink the lower bound estimate by 10% so the true lambda_min isn't underestimated-past
    lambda_max = maximum(ritz) * T(1.1) # grow the upper bound estimate by 10% so the true lambda_max isn't overestimated-past

    return lambda_min, lambda_max
end

# `degree` matvecs per preconditioner application; workspace vectors
# preallocated once (same lifetime as CGIterativeSolver) and reused every
# ldiv! call. lambda_min/lambda_max are refreshed once per *solve* by
# update_chebyshev_bounds! (not on every ldiv! call within that solve's CG
# run), since the operator only changes between solves, not within one.
"""
$(TYPEDSIGNATURES)

A GPU-capable, dot-product-free accelerated preconditioner for [`CGIterativeSolver`](@ref):
approximates `A^-1` by a degree-`degree` Chebyshev polynomial in the Jacobi-scaled operator
(Saad Algorithm 12.1), applied via `LinearAlgebra.ldiv!` below using only matvecs and
axpy-style updates -- no reductions (and so no GPU->CPU syncs) within one preconditioner
application. `lambda_min`/`lambda_max` (the polynomial's eigenvalue bounds) are refreshed once
per solve by [`update_chebyshev_bounds!`](@ref), not on every `ldiv!` call.

# Notes

Confirmed on real GPU hardware (see the module-level note above) that this does NOT eliminate
CG's own per-iteration host syncs (those come from CG's own dot products, not the preconditioner)
-- measured to lose to plain Jacobi in wall time at the grid sizes tested. Kept available as an
option (and the only GPU-capable *accelerated* one) since a differently-shaped or larger problem
could tip that balance; [`AMGPreconditioner`](@ref) is the better choice when available
(CPU/`SparseMatrixCSC`-only).
"""
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

"""
$(TYPEDSIGNATURES)

Builds a [`ChebyshevPreconditioner`](@ref) wrapping operator `A` with Jacobi diagonal `d`, degree
`degree` (`>= 1`), estimating eigenvalue bounds from `nsteps_estimate` (`>= 2`) Lanczos steps.
`lambda_min`/`lambda_max` start at placeholder value `1` (degenerates to plain Jacobi) until the
first real [`update_chebyshev_bounds!`](@ref) call.
"""
function ChebyshevPreconditioner(A, d::V, degree::Int; nsteps_estimate::Int = 15) where V <: AbstractVector
    T = eltype(d)
    degree >= 1 || error("ChebyshevPreconditioner: degree must be >= 1 (got $degree)")
    nsteps_estimate >= 2 || error("ChebyshevPreconditioner: nsteps_estimate must be >= 2 (got $nsteps_estimate)")
    op = JacobiScaledOperator(A, d)
    r, p, Ap = similar(d), similar(d), similar(d)
    return ChebyshevPreconditioner(op, degree, nsteps_estimate, one(T), one(T), r, p, Ap) # placeholder bounds, overwritten before first use
end

"""
$(TYPEDSIGNATURES)

Refreshes `P.lambda_min`/`P.lambda_max` from the current `rhs` via
[`estimate_eigenvalue_bounds`](@ref) -- cheap (`P.nsteps_estimate` matvecs) but real, since the
operator's spectrum shifts between Picard iterations/timesteps. Falls back to the previous
(valid) bounds if the estimate is numerically invalid (see the extensive in-code note for why
that can happen and why the fallback is always safe).
"""
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
        estimate_eigenvalue_bounds(P.op, rhs, P.nsteps_estimate) # re-run the short Lanczos estimate against this solve's current rhs
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow() # only swallow the tridiagonal-eigensolve failure mode; anything else is a real bug, propagate it
        (NaN, NaN) # sentinel: fails the isfinite check below, taking the same fallback path as a bogus-but-finite estimate
    end

    if isfinite(lambda_min) && isfinite(lambda_max) && lambda_min > 0 && lambda_max > lambda_min # sanity-check the estimate: finite, positive, and a proper (non-empty) interval
        P.lambda_min, P.lambda_max = lambda_min, lambda_max # estimate looks valid -- adopt it for this solve
    else
        @warn "ChebyshevPreconditioner: eigenvalue bound estimate was invalid (lambda_min=$lambda_min, lambda_max=$lambda_max); reusing previous bounds for this solve" maxlog = 10 # estimate looks bogus -- leave P.lambda_min/lambda_max untouched, reusing whatever was already there (construction-time placeholder or last valid solve's bounds)
    end

    return P # mutated in place; returned for chaining convenience
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
# gives near mesh-independent convergence at
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

"""
$(TYPEDSIGNATURES)

The default, fastest-converging preconditioner for `CGIterativeSolver{<:SparseAssembledLinearSystem}`:
Ruge-Stuben algebraic multigrid (AlgebraicMultigrid.jl), giving near mesh-independent CG iteration
counts (measured ~flat 5-6 iterations from 64x64 to 256x256, vs. Jacobi's 192->745 and
Chebyshev's 58->350 over the same grids -- see `test/benchmarks/`). CPU/`SparseMatrixCSC`-only
(AlgebraicMultigrid.jl has no GPU array support), so unlike [`ChebyshevPreconditioner`](@ref) this
isn't offered for `MatrixFreeLinearSystem`.
"""
mutable struct AMGPreconditioner{P}
    precond::P # AlgebraicMultigrid.jl's aspreconditioner(::MultiLevel) wrapper; already implements LinearAlgebra.ldiv!, so ldiv! below just delegates
end

"""
$(TYPEDSIGNATURES)

Builds an [`AMGPreconditioner`](@ref) hierarchy from `M`'s current values.

# Notes

`M`'s sparsity pattern is fixed, but its VALUES change every Picard iteration -- like
[`ChebyshevPreconditioner`](@ref)'s bounds, the AMG hierarchy (strength-of-connection,
aggregation, interpolation) is built from those values, so it has to be rebuilt every solve (see
[`update_amg!`](@ref)), not just once at construction.
"""
AMGPreconditioner(M::SparseMatrixCSC) = AMGPreconditioner(aspreconditioner(ruge_stuben(M)))

"""
$(TYPEDSIGNATURES)

Rebuilds `P`'s AMG hierarchy in place from `M`'s current values (not free -- see
[`AMGPreconditioner`](@ref)'s docstring for why this can't just be done once at construction).
"""
function update_amg!(P::AMGPreconditioner, M::SparseMatrixCSC)
    P.precond = aspreconditioner(ruge_stuben(M))
    return P
end

LinearAlgebra.ldiv!(y::AbstractVector, P::AMGPreconditioner, x::AbstractVector) = ldiv!(y, P.precond, x)
