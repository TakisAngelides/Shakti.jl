struct ModelParameters{F <: AbstractFloat, NE1, NE2, NE3}
    rho_w::F   # density of water
    rho_i::F   # density of ice
    g::F       # gravitational acceleration
    nu::F      # kinematic viscosity of water
    n::F       # Glen's flow law exponent
    omega::F   # parameter controlling transition from laminar to turbulent flow
    L::F       # latent heat of fusion
    br::F      # bedrock bump height
    lr::F      # bedrock bump spacing
    ct::F      # change of pressure melting point with temperature
    cw::F      # heat capacity of water
    p_atm::F   # atmospheric pressure, used as the Dirichlet reference for LAND/OCEAN BCs
    b_min::F   # minimum water thickness
    e_v::F     # englacial storage void ratio
    n_exp::NE1         # canonical_exponent(n), see the fast-exponentiation note below
    n_minus_1_exp::NE2 # canonical_exponent(n - 1)
    inv_n_exp::NE3     # canonical_exponent(1 / n)
end

function ModelParameters(;
    F::Type{<:AbstractFloat} = floattype,
    rho_w = 1000.0,
    rho_i = 910.0,
    g = 9.81,
    nu = 1.787e-6,
    n = 3.0,
    omega = 0.001,
    L = 334e3,
    br = 0.05,
    lr = 2.0,
    ct = 7.5e-8,
    cw = 4.22e3,
    p_atm = 0.0,
    b_min = 0.0,
    e_v = 0.0)

    n_F = F(n)
    n_exp = canonical_exponent(n_F)
    n_minus_1_exp = canonical_exponent(n_F - 1)
    inv_n_exp = canonical_exponent(1 / n_F)

    return ModelParameters(
        F(rho_w), F(rho_i), F(g), F(nu), n_F, F(omega), F(L), F(br), F(lr), F(ct), F(cw), F(p_atm), F(b_min), F(e_v),
        n_exp, n_minus_1_exp, inv_n_exp
    )

end

# =============================================================================
# Fast exponentiation for Glen's-flow-law-style exponents (n, n-1, 1/n, ...)
# =============================================================================
#
# `x^y` where `y::AbstractFloat` (even when y's *value* happens to be a whole
# number, e.g. `2.0`) always takes Julia's general floating-point-exponent
# path (conceptually `exp(y*log(x))`), which is markedly slower than the
# power-by-squaring path used for `x^y::Integer` -- and this can't be fixed
# by the compiler on its own, since `n` (ModelParameters' Glen's-law exponent)
# is a runtime value, not a literal, so Julia can't tell from its *type*
# alone (always some AbstractFloat) that its *value* happens to be integral.
#
# The fix is multiple dispatch on the exponent's type, decided ONCE -- not
# outside the hot per-cell loop/kernel only, but outside the whole time loop:
# ModelParameters' constructor above computes n_exp/n_minus_1_exp/inv_n_exp
# via canonical_exponent ONE time, when p is built, and stores each as its
# own field (typed NE1/NE2/NE3 -- Int or F, whichever canonical_exponent
# returned). Every `compute_xxx!` wrapper (compute_taub_x!, compute_b!, ...)
# then just reads e.g. `p.n_minus_1_exp` -- no `isinteger` check at all, not
# even once per call, since p.n never changes after construction:
#
#   pow(abs(N[i, j]), p.n_minus_1_exp)   # dispatches on the field's TYPE, decided at ModelParameters construction
#
# `canonical_exponent` converts a whole-valued Float (`3.0`) to a plain `Int`
# (`3`); `pow` then dispatches on that Int vs Float distinction. For the
# common case of an integer Glen's-law exponent (n=3 is standard), this
# means every `pow(abs(N), 2)` (n-1) inside the hot assembly/melt/creep
# kernels takes the fast integer path automatically, while a genuinely
# fractional `n` (e.g. 2.5) still works correctly, just without the
# speedup -- pow's fallback method is exactly `x^y` either way.
"""
    canonical_exponent(n)

Returns `n` converted to a plain `Int` if it has an exact integer value
(e.g. `3.0 -> 3`), otherwise returns `n` unchanged. See the module-level
note above for why/how this is used.
"""
canonical_exponent(n::AbstractFloat) = isinteger(n) ? Int(n) : n
canonical_exponent(n::Integer) = n

"""
    pow(x, n)

`x^n`, dispatching on `n`'s type: `n::Integer` takes Julia's fast
power-by-squaring path; `n::AbstractFloat` falls back to the general (much
slower) floating-point-exponent path. Meant to be called with an exponent
that has already been through `canonical_exponent` -- see the module-level
note above.
"""
@inline pow(x, n::Integer) = x^n
@inline pow(x, n::AbstractFloat) = x^n