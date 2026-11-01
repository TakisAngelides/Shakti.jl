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
# iteration. What it does cut is the NUMBER of CG iterations, and with them
# the syncs. Measured 2026-11-01 (MatrixFreeLinearSystem, multigrid = false,
# degree 4, default CG tolerance, test/benchmarks/benchmark_solver.jl's
# "simple" setup, 10 steps of 1 h, 131-141 solves): Chebyshev needs 3.6-3.8x
# fewer CG iterations than Jacobi (53/108/215/420 vs 190/405/808/1587 per
# solve at 64x64/128x128/256x256/512x512). On an A100 (1 thread) that makes
# it 2.2x faster than Jacobi in wall time at every size (2.8 vs 6.1 s at
# 64x64, 21.8 vs 47.2 s at 512x512): a Jacobi-preconditioned CG iteration
# costs a roughly flat ~215-230us from 64x64 to 512x512 (fixed per-iteration
# overhead, kernel launches + host syncs, dominating device compute), a
# Chebyshev one ~360-400us, which the 3.7x fewer iterations more than repay.
# On the CPU (8 threads) the four extra matvecs per application are not
# cheap, and Chebyshev only ties Jacobi (59.3 vs 56.5 s at 256x256, 368 vs
# 387 s at 512x512). An earlier measurement (64x64/128x128 on an A100) had
# Jacobi 12%/29% faster than Chebyshev; it predates the second-step fix in
# ldiv! below, but that fix is worth only ~8% in iterations, and the old
# recurrence too is 2x faster than Jacobi in the run above, so the earlier
# result does not reproduce for a reason other than the fix. Genuine
# sync-avoidance on GPU would need replacing CG itself with a dot-product-free
# outer solver (true Chebyshev semi-iteration AS the solver, not just the
# preconditioner).
#
# Net effect: for CGIterativeSolver{<:SparseAssembledLinearSystem} (CPU-only),
# AMGPreconditioner (below) is the better choice and is the default there. For
# MatrixFreeLinearSystem (the GPU-capable path, where AMGPreconditioner isn't
# available -- AlgebraicMultigrid.jl has no GPU array support), the default is
# AggregationMultigrid (see multigrid.jl), and it beats both on GPU by a margin
# that grows with the grid: in the same A100 run, 5.8/7.2/8.1/8.9 CG iterations
# from 64x64 to 512x512, and 1.2/1.5/1.8/2.5 s, 2.3/3.4/5.6/8.8x faster than
# degree-4 Chebyshev and 5-19x faster than Jacobi. Chebyshev is only the better
# of the two multigrid = false options there.
#
# Applies the diagonal-Jacobi-scaled operator D^-1*A. The same wrapper works
# for both SparseAssembledLinearSystem's SparseMatrixCSC and
# MatrixFreeLinearSystem's StencilOperator, since both already have a mul!
# method -- nothing here is backend- or representation-specific.
"""
$(TYPEDSIGNATURES)

Applies the diagonal-Jacobi-scaled operator `D^-1*A` via `mul!` below. The same wrapper works for
both `SparseAssembledLinearSystem`'s `SparseMatrixCSC` and `MatrixFreeLinearSystem`'s
`StencilOperator`, since both already have a `mul!` method -- nothing here is backend- or
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

`r`/`p`/`Ap` are scratch buffers of the same length as `rhs`, overwritten in place -- callers
pass [`ChebyshevPreconditioner`](@ref)'s own preallocated `r`/`p`/`Ap` fields (safe to clobber:
by the time `ldiv!` later reads them, this function has long since returned and they hold no
state it needs). This avoids allocating three fresh length-`N` vectors every solve just to
throw them away -- at 512x512 this alone was ~6.2MB/call, the largest single avoidable
allocation found in the 2026-09-22 benchmark sweep (`PostDoc_Latex_Notes/Notes_on_Shakti/
linear_solver_benchmarks.tex`, Section 5.2).
"""
function estimate_eigenvalue_bounds!(r::AbstractVector, p::AbstractVector, Ap::AbstractVector, op::JacobiScaledOperator, rhs::AbstractVector, nsteps::Int)

    T = eltype(rhs) # scalar type to match rhs (Float32/Float64), so all work below stays in that precision

    r .= rhs # r_0 = rhs - op*0 = rhs (x0 = 0)
    p .= r   # p_0 = r_0, the initial CG search direction
    # Ap is pure scratch (written by mul! before ever being read) -- no initial value needed

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
    gershgorin::Bool # true (default): fixed bounds from Gershgorin's theorem, see update_chebyshev_bounds!; false: the Lanczos estimate
    eig_ratio::T     # lambda_min = lambda_max/eig_ratio under gershgorin = true
end

"""
$(TYPEDSIGNATURES)

Builds a [`ChebyshevPreconditioner`](@ref) wrapping operator `A` with Jacobi diagonal `d`, degree
`degree` (`>= 1`), estimating eigenvalue bounds from `nsteps_estimate` (`>= 2`) Lanczos steps.
`lambda_min`/`lambda_max` start at placeholder value `1` (degenerates to plain Jacobi) until the
first real [`update_chebyshev_bounds!`](@ref) call.
"""
function ChebyshevPreconditioner(A, d::V, degree::Int; nsteps_estimate::Int = 15, bounds::Symbol = :gershgorin, eig_ratio = 30) where V <: AbstractVector
    T = eltype(d)
    bounds in (:gershgorin, :lanczos) || error("ChebyshevPreconditioner: bounds must be :gershgorin or :lanczos (got $bounds)")
    eig_ratio > 1 || error("ChebyshevPreconditioner: eig_ratio must be > 1 (got $eig_ratio)")
    degree >= 1 || error("ChebyshevPreconditioner: degree must be >= 1 (got $degree)")
    nsteps_estimate >= 2 || error("ChebyshevPreconditioner: nsteps_estimate must be >= 2 (got $nsteps_estimate)")
    op = JacobiScaledOperator(A, d)
    r, p, Ap = similar(d), similar(d), similar(d)
    return ChebyshevPreconditioner(op, degree, nsteps_estimate, one(T), one(T), r, p, Ap, bounds == :gershgorin, T(eig_ratio)) # placeholder bounds, overwritten before first use
end

"""
$(TYPEDSIGNATURES)

Refreshes `P.lambda_min`/`P.lambda_max` from the current `rhs` via
[`estimate_eigenvalue_bounds!`](@ref) -- cheap (`P.nsteps_estimate` matvecs) but real, since the
operator's spectrum shifts between Picard iterations/timesteps. Falls back to the previous
(valid) bounds if the estimate is numerically invalid (see the extensive in-code note for why
that can happen and why the fallback is always safe).
"""
function update_chebyshev_bounds!(P::ChebyshevPreconditioner, rhs::AbstractVector)

    # Default: bounds straight from Gershgorin's theorem, no matvecs at all. Every assembled row
    # has a diagonal at least as large as the sum of its off-diagonal magnitudes (diffusion +
    # non-negative reaction; Dirichlet/frozen rows are identity rows), so every eigenvalue of the
    # Jacobi-scaled operator D^-1*A lies in (0, 2] -- lambda_max = 2 is a guaranteed upper bound.
    # That guarantee is what matters: an interval whose top lies BELOW the true lambda_max makes
    # the Chebyshev polynomial blow up on the top of the spectrum, the preconditioner indefinite,
    # and CG stall -- which the short Lanczos estimate below (started from the smooth rhs, so it
    # can miss the high end) was measured to cause on real glacier grids (Helheim, Drang Drung:
    # wrong final N by 3-59%). lambda_min = lambda_max/eig_ratio is the standard Chebyshev
    # choice; CG itself resolves the modes below it.
    if P.gershgorin
        T = typeof(P.lambda_max)
        P.lambda_max = T(2)
        P.lambda_min = T(2) / P.eig_ratio
        return P
    end

    # The short, unreorthogonalized Lanczos recurrence in estimate_eigenvalue_bounds!
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
        # Reuses P's own r/p/Ap scratch fields (otherwise idle until ldiv! runs later in this
        # same solve) instead of allocating three fresh length-N vectors just for this estimate.
        estimate_eigenvalue_bounds!(P.r, P.p, P.Ap, P.op, rhs, P.nsteps_estimate) # re-run the short Lanczos estimate against this solve's current rhs
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

    # The second step is special because T_1(t) = t, not 2t: beta_1 = (c*alpha_1)^2/2, then
    # beta_k = (c*alpha_k/2)^2 for k >= 2 (Gutknecht & Roellin 2002, correcting the Templates
    # book's Fig. 2.11). Using (c*alpha/2)^2 throughout gives a residual polynomial that is
    # not the Chebyshev one: max|1 - lambda*p(lambda)| 0.61 instead of 0.43 at degree 4 on [2/30, 2].
    alpha_prev = alpha
    for k in 2:P.degree
        beta = k == 2 ? (c_rad * alpha_prev)^2 / 2 : (c_rad * alpha_prev / 2)^2
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
# gives near mesh-independent convergence: 4.5/5.4/6.0/6.8 mean CG
# iterations per solve at 64x64/128x128/256x256/512x512, essentially flat
# despite 64x more unknowns, vs Jacobi's 190->1587 and Chebyshev(degree 4)'s
# 53->420 over the same grids (2026-11-01, 8 threads, default CG tolerance,
# test/benchmarks/benchmark_solver.jl's "simple" setup, 10 steps of 1 h).
# That makes AMG substantially faster in wall time once the grid is large
# enough for its (nontrivial -- see below) hierarchy setup cost to be worth
# it: even rebuilding the hierarchy every solve (refresh_every = 1), it ties
# Jacobi and Chebyshev at 64x64 (1.1-1.2 s), is ~4.2x faster than both at
# 256x256 (19.0 s vs 80.2/81.0 s) and ~10x at 512x512 (72 s vs 700/766 s).
#
# CPU/SparseMatrixCSC-only (AlgebraicMultigrid.jl has no GPU array support),
# so -- like CholeskyDirectSolver -- this is only offered for
# CGIterativeSolver{<:SparseAssembledLinearSystem}, not MatrixFreeLinearSystem.
#
# `refresh_every` (below): the 2026-09-22 benchmark report (linear_solver_benchmarks.tex,
# Section 5.2/6.1) found the from-scratch hierarchy rebuild is by far the single largest
# allocator measured anywhere in that whole study (up to ~8.6GB/call at 2048x2048) -- and, per
# Section 6.1's discussion, it's rebuilt every solve for no numerical necessity: AMG's classical
# advantage assumes its setup cost is amortized over MANY solves against a fixed operator, but
# here it's paid once and used once. `refresh_every > 1` amortizes it properly: skip the
# rebuild for `refresh_every - 1` calls out of every `refresh_every`, reusing the existing
# hierarchy. This is NOT the same as reusing a fully stale preconditioner: AlgebraicMultigrid.jl's
# `ruge_stuben` stores the finest level's matrix (and its Gauss-Seidel smoothers) as a direct
# REFERENCE to whatever `SparseMatrixCSC` is passed in (confirmed by reading
# AlgebraicMultigrid.jl v2.0.1's own source, src/classical.jl's `extend_hierarchy_rs!` and
# src/smoother.jl's `FastGSSmoother`/`gs!`, which read `A.nzval` live) -- and `sals.M` is the
# exact same array object mutated in place every Picard iteration (never reallocated), so the
# finest-level smoothing and residual evaluation see CURRENT values even when the hierarchy
# itself isn't rebuilt. Only the coarser levels (interpolation/restriction operators and their
# Galerkin-product matrices, a one-time snapshot from whichever solve last rebuilt them) go
# stale between refreshes. Trades some CG iteration-count growth between refreshes for a large
# cut in allocation and wall time; `refresh_every = 1` is the original, always-rebuild behavior.
#
# Default `refresh_every = 5`, measured at 256x256 (8 threads, filled sheet, 2026-11-01): one
# rebuild costs ~45 ms and ~133 MB against ~63 ms for the CG solve it preconditions (~7
# iterations), i.e. ~47% of setup+solve. A hierarchy one Picard iterate old costs ~0.2 extra CG
# iterations; one time step old ~5 (up to 32 at a step change), so counting solves beats
# refreshing once per time step. Over a 15-step run (test/benchmarks/amg_refresh_sweep.jl,
# steps 2-15), refresh_every = 1/2/5/10 gave 5.04/5.15/5.40/5.73 mean CG iterations and
# 142/119/83/77 ms per solve; 5 leaves margin on badly conditioned glacier grids.
#
# The construction-time hierarchy is built from SparseAssembledLinearSystem's placeholder
# matrix (identity diagonal, zero off-diagonals, so a hierarchy with no coarse levels at all);
# it only fixes the field's type. The counter therefore starts at `refresh_every - 1`, so the
# first `update_amg!` call always rebuilds from the first assembled matrix -- starting it at 0
# left the first `refresh_every - 1` solves effectively unpreconditioned (~1050 CG iterations
# instead of ~9 at 256x256 in the 2026-09-22 sweep).

"""
$(TYPEDSIGNATURES)

The default, fastest-converging preconditioner for `CGIterativeSolver{<:SparseAssembledLinearSystem}`:
Ruge-Stuben algebraic multigrid (AlgebraicMultigrid.jl), giving near mesh-independent CG iteration
counts (measured ~flat 4.5-6.8 iterations from 64x64 to 512x512, vs. Jacobi's 190->1587 and
degree-4 Chebyshev's 53->420 over the same grids -- see the note above). CPU/`SparseMatrixCSC`-only
(AlgebraicMultigrid.jl has no GPU array support), so unlike [`ChebyshevPreconditioner`](@ref) this
isn't offered for `MatrixFreeLinearSystem`.
"""
mutable struct AMGPreconditioner{P}
    precond::P # AlgebraicMultigrid.jl's aspreconditioner(::MultiLevel) wrapper; already implements LinearAlgebra.ldiv!, so ldiv! below just delegates
    refresh_every::Int # rebuild the hierarchy every this many update_amg! calls; 1 = always (original behavior). See the module note above.
    calls_since_refresh::Int # counter driving the above; starts at refresh_every - 1 so the first update_amg! call rebuilds (the construction-time hierarchy comes from the placeholder matrix)
end

"""
$(TYPEDSIGNATURES)

Builds an [`AMGPreconditioner`](@ref) hierarchy from `M`'s current values.

# Notes

`M`'s sparsity pattern is fixed, but its VALUES change every Picard iteration -- like
[`ChebyshevPreconditioner`](@ref)'s bounds, the AMG hierarchy (strength-of-connection,
aggregation, interpolation) is built from those values, so it has to be rebuilt as they drift
(see [`update_amg!`](@ref)), not just once at construction. `refresh_every` (default `5`)
rebuilds it on the first [`update_amg!`](@ref) call and then every `refresh_every`-th call,
reusing it in between (the lagged-hierarchy amortization described in the module note above);
`refresh_every = 1` rebuilds on every call.
"""
function AMGPreconditioner(M::SparseMatrixCSC; refresh_every::Int = 5)
    refresh_every >= 1 || error("AMGPreconditioner: refresh_every must be >= 1 (got $refresh_every)")
    return AMGPreconditioner(aspreconditioner(ruge_stuben(M)), refresh_every, refresh_every - 1) # counter primed so the first update_amg! rebuilds
end

"""
$(TYPEDSIGNATURES)

Advances `P`'s refresh counter and rebuilds `P`'s AMG hierarchy in place from `M`'s current
values on the first call and then once every `P.refresh_every` calls (every call, if
`P.refresh_every == 1`) -- see [`AMGPreconditioner`](@ref)'s docstring for why this can't just be done once at
construction, and the module-level note above for why skipping a rebuild is still safe/sensible,
not merely cheap).
"""
function update_amg!(P::AMGPreconditioner, M::SparseMatrixCSC)
    P.calls_since_refresh += 1
    if P.calls_since_refresh >= P.refresh_every
        P.precond = aspreconditioner(ruge_stuben(M))
        P.calls_since_refresh = 0
    end
    return P
end

LinearAlgebra.ldiv!(y::AbstractVector, P::AMGPreconditioner, x::AbstractVector) = ldiv!(y, P.precond, x)
