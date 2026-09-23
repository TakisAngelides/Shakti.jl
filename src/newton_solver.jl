# Jacobian-Free Newton-Krylov (JFNK) solver for the same nonlinear elliptic head equation
# PicardSolver (elliptic_solver.jl) solves, as an alternative outer nonlinear iteration.
#
# THEORY (see beyond_solver_choice.tex, or the notes on why this is being tried at all): Picard
# iteration h_{k+1} = g(h_k) converges only LINEARLY -- the error shrinks by a roughly constant
# factor per iteration, however small that factor is. Newton's method, applied to the actual
# nonlinear residual, converges QUADRATICALLY near the solution. The residual Picard's own fixed
# point satisfies is exactly
#
#     F(h) := A(h) h - b(h) = 0
#
# where A(h)/b(h) are precisely what update_SALS_elliptic! builds (freeze every h-dependent
# coefficient -- K, N, the Newton-linearized creep-closure term, sliding-law inputs -- AT h, then
# assemble the resulting locally-linear system). F(h)=0 iff h is the Picard fixed point, i.e. iff
# h solves the true discretized nonlinear equation Picard is designed to converge to -- so driving
# F(h) to zero via Newton's method targets the same solution, just via a different, faster-
# converging iteration.
#
# Newton's method needs the Jacobian dF/dh, which is NOT just A(h) (that's what Picard uses as an
# approximation to it, missing the dA/dh*h - db/dh contributions from every coefficient's own
# h-dependence) -- deriving it analytically would mean differentiating K(h), N(h), the sliding law,
# and every melt term by hand, a large and error-prone undertaking across many physics modules.
# Jacobian-Free Newton-Krylov (JFNK; see e.g. Knoll & Keyes, "Jacobian-free Newton-Krylov methods:
# a survey of approaches and applications", J. Comput. Phys. 193(2), 2004) sidesteps this
# entirely: it only ever needs Jacobian-VECTOR products J(h)*v, which can be approximated by a
# finite-difference directional derivative of F itself,
#
#     J(h) v ≈ (F(h + εv) - F(h)) / ε,             ε = sqrt(eps_machine) * (1 + ||h||) / ||v||
#
# (the standard JFNK step-size formula, balancing floating-point roundoff in the numerator against
# the finite-difference truncation error -- see Knoll & Keyes Eq. 14) -- i.e. it only ever needs
# MORE EVALUATIONS of the exact same residual function Picard already computes, never an explicit
# Jacobian matrix. This is exactly the same "matrix-free" spirit as MatrixFreeLinearSystem's own
# stencil-vector product for CG, just one level up (accelerating the OUTER nonlinear loop, not an
# inner linear solve).
#
# Each Newton step then solves J(h_k) Δ ≈ -F(h_k) approximately (an "inexact Newton" step -- GMRES
# is run to a loose tolerance, not full precision, especially far from the solution where an exact
# linear solve would be wasted effort) via Krylov.jl's GMRES (not CG: J is not symmetric in
# general, even though A(h) alone is SPD). This inner GMRES solve is PRECONDITIONED by
# A(h_k)'s own Cholesky factorization -- confirmed necessary, not optional, by direct measurement:
# an unpreconditioned version worked fine at 32x32 (converging in as few as 2 outer Newton
# iterations per timestep, the expected quadratic-convergence signature) but completely failed to
# converge at 256x256 (hit the GMRES iteration cap every single outer iteration, all 50 allowed
# outer iterations, every timestep) -- exactly the same qualitative failure mode plain-Jacobi CG
# has without AMG (linear_solver_benchmarks.tex): an unpreconditioned Krylov method's iteration
# count for this elliptic-type operator grows badly with grid size. A(h_k) is a natural, "free"
# preconditioner for J(h_k): it IS the Picard linearization of the same equation at the same point
# (missing only the dA/dh, db/dh terms Newton's true Jacobian includes), and Shakti already has to
# assemble and factorize it anyway to evaluate the residual F(h_k) -- reusing that factorization as
# a preconditioner costs nothing extra beyond what a single Picard iteration would already pay.
# Then h_{k+1} = h_k + α Δ, where α ∈ (0, 1] is a simple
# backtracking line-search factor (halved until ||F(h_k + αΔ)|| < ||F(h_k)||, or a minimum step is
# hit) -- pure undamped Newton steps are well known to diverge far from the solution, and Shakti's
# real datasets have already shown real convergence difficulty for a much gentler iteration
# (Picard's own b_max clamp exists specifically because of a real runaway on Drang Drung).

"""
$(TYPEDSIGNATURES)

Evaluates the nonlinear elliptic-head residual `F(h) = A(h) h - b(h)` at `h_vec` (flat, length
`g.nx*g.ny`), writing it into `res`. `A(h)`/`b(h)` are exactly [`update_SALS_elliptic!`](@ref)'s
`sals.M`/`sals.rhs`, rebuilt from `h_vec` via [`refresh_head_dependents!`](@ref) first (so every
h-dependent coefficient -- `K`, `N`, `mdot`, ... -- is refreshed at the CURRENT `h_vec`, not
whatever `s.h` held before this call). Mutates `s.h` in place to `h_vec` as a side effect (needed
so `refresh_head_dependents!`/`update_SALS_elliptic!` see it) -- callers that still need the
PREVIOUS `s.h` value afterward must save it first.
"""
function elliptic_residual!(res::AbstractVector, h_vec::AbstractVector, sals::SparseAssembledLinearSystem,
                             s::State, g::Grid, p::ModelParameters, mt::MeltTerms, kfs::AbstractKFaceScheme,
                             sl::AbstractSlidingLaw; cnc::AbstractCellNClamping = NoCellNClamping(),
                             ds::AbstractDiffusionScheme = NoDiffusion())
    s.h .= reshape(h_vec, g.nx, g.ny)
    refresh_head_dependents!(s, g, p, mt, kfs, sl; cnc, ds)
    update_SALS_elliptic!(sals, s, g, p, kfs, ds)
    mul!(res, sals.M, h_vec)
    res .-= sals.rhs
    return res
end

"""
$(TYPEDSIGNATURES)

Matrix-free Jacobian operator for [`elliptic_residual!`](@ref) at a fixed base point `h0`/`r0 =
F(h0)`: `mul!(y, op, v)` computes the finite-difference Jacobian-vector product `J(h0) v` (see the
module-level notes above for the formula/theory). `h0`/`r0` are set once per outer Newton
iteration (`Newton_iteration!`) and stay fixed across every GMRES inner iteration that iteration
performs -- `v` (and therefore the finite-difference step size `ε`, which depends on `norm(v)`)
changes every `mul!` call, `h0`/`r0` do not.
"""
mutable struct JFNKOperator{F <: AbstractFloat, V <: AbstractVector{F}, SALS <: SparseAssembledLinearSystem,
                             S <: State, G <: Grid, P <: ModelParameters, MT <: MeltTerms,
                             KFS <: AbstractKFaceScheme, SL <: AbstractSlidingLaw,
                             CNC <: AbstractCellNClamping, DS <: AbstractDiffusionScheme}
    sals::SALS
    state::S
    grid::G
    p::P
    mt::MT
    kfs::KFS
    sl::SL
    cnc::CNC
    ds::DS
    h0::V     # base point for this Newton iteration (aliases the Newton solver's own h buffer -- read-only here)
    r0::V     # F(h0), precomputed once per Newton iteration
    h_pert::V # workspace: h0 + ε*v
    r_pert::V # workspace: F(h0 + ε*v)
    n::Int
end

Base.eltype(::JFNKOperator{F}) where F = F
Base.size(op::JFNKOperator) = (op.n, op.n)
Base.size(op::JFNKOperator, i::Int) = size(op)[i]

function LinearAlgebra.mul!(y::AbstractVector, op::JFNKOperator, v::AbstractVector)
    F = eltype(op)
    nv = norm(v)
    if nv < eps(F)
        fill!(y, zero(F)) # a zero (or numerically negligible) direction has no meaningful derivative to approximate
        return y
    end
    eps_fd = sqrt(eps(F)) * (1 + norm(op.h0)) / nv # standard JFNK step size (Knoll & Keyes 2004, Eq. 14)
    @. op.h_pert = op.h0 + eps_fd * v
    elliptic_residual!(op.r_pert, op.h_pert, op.sals, op.state, op.grid, op.p, op.mt, op.kfs, op.sl; op.cnc, op.ds)
    @. y = (op.r_pert - op.r0) / eps_fd
    return y
end

"""
$(TYPEDSIGNATURES)

Drives the same nonlinear elliptic head equation [`PicardSolver`](@ref)/[`Picard_loop!`](@ref)
solves, via Jacobian-Free Newton-Krylov instead of Picard iteration -- see the module-level notes
at the top of this file for the full theory. `gmres_rtol`/`gmres_itmax` control the INNER
(linear, per-Newton-step) GMRES solve -- an inexact Newton method deliberately solves this loosely,
especially early on, rather than to full precision. `damping_min` is the smallest backtracking
line-search step-length factor tried before giving up and accepting whatever step was found (never
skipping the line search entirely -- pure undamped Newton is not safe on Shakti's real, sometimes
stiff problems). Convergence is checked on the residual directly (`norm(F(h))`, relative to
`norm(b(h))`, i.e. the current RHS `sals.rhs` -- NOT `norm(h)`, an earlier version of this check
that turned out to be dimensionally wrong: `F(h) = A(h)h - b(h)` lives in the same units as `b(h)`,
not in `h`'s own units, and on the real Greenland dataset (much larger `h`/coefficient magnitudes
than the synthetic/Drang-Drung grids this was first validated on) that mismatch let the check pass
trivially, at timestep 1, against the raw un-relaxed initial condition -- a silent false positive:
Newton never took a single real step for the entire run, while Cholesky needed 39 genuine Picard
iterations from that same starting point. See `project_shakti_performance_findings.md`/session
notes for the full failure analysis this fix responds to) unlike [`PicardSolver`](@ref)'s
update-based check, since Newton's own step size is not otherwise available before the first step
is taken -- one consequence, seen directly in real testing (below): a timestep whose previous
solution ALREADY satisfies the new timestep's tolerance converges in 0 outer iterations under this
check, something Picard's own convergence criterion structurally cannot report (it requires at
least one completed iteration before it can measure an update size at all), so raw outer-iteration
counts between the two are not perfectly apples-to-apples.

**Empirical results (first validation pass, `beyond_solver_choice.tex`/session notes)**: on the
synthetic benchmark (uniform slope/coefficients, `test/benchmarks/newton_vs_picard.jl`) at
256x256, Newton (preconditioned as described above) converges in ~2-3 outer Newton iterations per
timestep (1-2 inner GMRES iterations each) vs Picard's ~9-11 Picard iterations -- a genuine
**42.5% wall-time reduction** (6.54s vs 11.36s over 15 timesteps). On the real Drang Drung v2
dataset (`test/drangdrung/drangdrung_newton_compare.jl`, 30 real timesteps): 28/30 timesteps
converged, most in 0 outer iterations (the near-steady-state case above) and dramatically faster
than Picard overall when they do -- but **2 consecutive timesteps (16, 17) hit the `iters` cap
without converging to `tol`**, each costing the full iteration budget (~1.45-1.48s), before the
simulation recovered cleanly on the very next timestep (no divergence, no NaN/blowup -- just an
under-converged pair of timesteps). This is consistent with a known, general weakness of Newton's
method: it relies on the local linearization being a reasonably good model of the true nonlinear
residual, which real non-smooth features (Shakti's own `b_max`/`CreepCutoff`-style clamps) can
locally violate in a way Picard's gentler, damped iteration tolerates better even though it's
slower on average. **Net assessment**: a real, substantial performance win when it converges, with
a real, not-yet-hardened robustness gap on genuinely stiff real transients -- opt-in only, not
recommended as a default without either a hybrid Picard fallback for the cap-hit case or further
robustness work (trust-region-style step control instead of plain backtracking, for instance).
"""
mutable struct NewtonJFNKSolver{F <: AbstractFloat, SALS <: SparseAssembledLinearSystem, FACT, WS, V <: AbstractVector{F}} <: AbstractEllipticSolver
    iters::Int
    tol::F
    gmres_rtol::F
    gmres_itmax::Int
    damping_min::F
    sals::SALS
    fact::FACT # Cholesky factorization of A(h_k), refreshed once per outer Newton iteration and used
               # as GMRES's preconditioner for the whole of that iteration's inner solve -- see the
               # module-level notes above for why this specific choice of preconditioner is natural
               # and effectively free (Shakti already assembles/factorizes it to evaluate F(h_k)).
    ws::WS # Krylov.jl GmresWorkspace
    converged::Bool
    last_iter::Int
    h0::V       # STABLE COPY of the current Newton base point h_k -- deliberately NOT a view of s.h,
                # since every JFNKOperator mul! call internally mutates s.h (via elliptic_residual!)
                # to evaluate a perturbed point; if this were an alias of s.h, the "fixed" base point
                # would silently drift with every finite-difference evaluation inside GMRES.
    r::V        # F(h_k), the current residual (cached across iterations to avoid recomputing)
    neg_r::V    # -F(h_k), the actual GMRES right-hand side
    h_trial::V  # workspace for a line-search trial point
    r_trial::V  # F(h_trial)
end

"""
$(TYPEDSIGNATURES)

Builds a [`NewtonJFNKSolver`](@ref) on grid `g`. `iters`/`tol` mirror [`PicardSolver`](@ref)'s
(max outer Newton iterations, convergence tolerance on the relative residual norm).
`gmres_memory` sets the GMRES Krylov subspace size before a restart.
"""
function NewtonJFNKSolver(g::Grid{F}; iters::Int = 50, tol = 1e-6, gmres_rtol = 1e-2,
                           gmres_itmax::Int = 30, damping_min = 1e-3, gmres_memory::Int = 20) where F
    n = g.nx * g.ny
    sals = SparseAssembledLinearSystem(g)
    fact = cholesky(Symmetric(sals.M)) # placeholder structure at construction (real values filled in before first use), same convention as CholeskyDirectSolver
    ws = GmresWorkspace(n, n, Vector{F}; memory = gmres_memory)
    h0      = @zeros(n)
    r       = @zeros(n)
    neg_r   = @zeros(n)
    h_trial = @zeros(n)
    r_trial = @zeros(n)
    return NewtonJFNKSolver(iters, F(tol), F(gmres_rtol), gmres_itmax, F(damping_min), sals, fact, ws,
                             false, 0, h0, r, neg_r, h_trial, r_trial)
end

"""
$(TYPEDSIGNATURES)

Solves the nonlinear elliptic head equation via [`NewtonJFNKSolver`](@ref) -- the
[`NewtonJFNKSolver`](@ref) counterpart to [`elliptic_solver!`](@ref)/[`Picard_loop!`](@ref).
"""
function elliptic_solver!(ns::NewtonJFNKSolver, state::State, grid::Grid, p::ModelParameters, mt::MeltTerms,
                           kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw; cnc::AbstractCellNClamping = NoCellNClamping(),
                           ds::AbstractDiffusionScheme = NoDiffusion())
    Newton_loop!(ns, state, grid, p, mt, kfs, sl; cnc, ds)
end

"""
$(TYPEDSIGNATURES)

Repeatedly takes inexact, line-searched Newton steps (up to `ns.iters` times), checking
convergence via the relative residual norm (`norm(F(h)) / (norm(b(h)) + eps) < ns.tol`, `b(h)` =
`sals.rhs` = the current RHS of `F(h) = A(h)h - b(h)` -- dimensionally the correct thing to
normalize against, unlike an earlier version of this check that normalized by `norm(h)` instead;
see the constructor's docstring above for why that was a real, silently-wrong-answer bug on the
real Greenland dataset), and sets `ns.converged`/`ns.last_iter` accordingly. See the module-level
notes for the full method.
"""
function Newton_loop!(ns::NewtonJFNKSolver, state::State, grid::Grid, p::ModelParameters, mt::MeltTerms,
                       kfs::AbstractKFaceScheme, sl::AbstractSlidingLaw; cnc::AbstractCellNClamping = NoCellNClamping(),
                       ds::AbstractDiffusionScheme = NoDiffusion())
    s = state
    g = grid
    h_vec = vec(s.h) # aliases s.h -- writes through this vector ARE writes to s.h

    ns.converged = false
    ns.last_iter = 0

    elliptic_residual!(ns.r, h_vec, ns.sals, s, g, p, mt, kfs, sl; cnc, ds)

    for iter in 1:ns.iters

        r_norm = norm(ns.r)
        b_norm = norm(ns.sals.rhs) # dimensionally matches r_norm (both live in F(h)=A(h)h-b(h)'s units); norm(h) does not
        if r_norm / (b_norm + eps(eltype(h_vec))) < ns.tol
            ns.converged = true
            ns.last_iter = iter - 1
            return
        end

        ns.h0 .= h_vec # snapshot the current Newton iterate BEFORE any residual evaluation can mutate s.h out from under it
        # ns.sals.M still holds A(h_k) here (nothing has evaluated the residual at a different point
        # since either the initial call above or the previous iteration's line search settled on
        # h_k) -- refactorize the PRECONDITIONER's own storage from it now, before op's finite
        # differences start overwriting ns.sals.M with A(perturbed points).
        cholesky!(ns.fact, Symmetric(ns.sals.M))
        op = JFNKOperator(ns.sals, s, g, p, mt, kfs, sl, cnc, ds, ns.h0, ns.r,
                           similar(h_vec), similar(ns.r), length(h_vec))

        @. ns.neg_r = -ns.r
        gmres!(ns.ws, op, ns.neg_r; M = ns.fact, ldiv = true, rtol = ns.gmres_rtol, itmax = ns.gmres_itmax)
        delta = ns.ws.x # Newton step, from solving J(h_k) Δ ≈ -F(h_k)

        # Backtracking line search: halve the step until the residual actually decreases, or give up at damping_min.
        # Uses ns.h0 (the stable snapshot), NOT h_vec/s.h -- gmres!'s internal JFNKOperator calls have
        # already left s.h in an arbitrary perturbed state from their own finite-difference evaluations.
        damping = one(eltype(h_vec))
        while true
            @. ns.h_trial = ns.h0 + damping * delta
            elliptic_residual!(ns.r_trial, ns.h_trial, ns.sals, s, g, p, mt, kfs, sl; cnc, ds)
            (norm(ns.r_trial) < r_norm || damping <= ns.damping_min) && break
            damping /= 2
        end

        h_vec .= ns.h_trial # writes through to s.h
        ns.r .= ns.r_trial
        ns.last_iter = iter

    end

    ns.last_iter = ns.iters
    return
end

"""
$(TYPEDSIGNATURES)

Mirrors [`picard_status`](@ref) for [`NewtonJFNKSolver`](@ref): returns `(converged, last_iter)`.
"""
newton_status(ns::NewtonJFNKSolver) = (ns.converged, ns.last_iter)
