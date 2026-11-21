# The test sheet of both GPU test files: SHMIP's square-root ice sheet (100 km x 20 km, land margin at x = 0) at 1 km
# (101 x 21 cells), SHMIP's constants, uniform input at SHMIP A3's rate, run for a few 6 h steps. `:default` is the
# default sheet (implicit upwind faces, a non-symmetric head matrix); `:spd` is SHAKTI's original filled sheet with
# arithmetic faces, whose head matrix is symmetric positive definite, for Cholesky and CG.
using Shakti, Test, LinearAlgebra, Serialization

const RESULTS = mkpath(joinpath(@__DIR__, "results"))
const SPD_SHEET = (cavity_filling = FilledCavities(), uplift = NoUplift(), hydraulic_potential = BedPotential(),
                   gap_coupling = ExplicitGapCoupling(), k_face_choice = "arithmetic")

function test_sheet(make_ls, sheet::Symbol; nsteps = 8, dt = 6 * 3600.0)
    lx, ly, dx = 100e3, 20e3, 1e3
    nx, ny = round(Int, lx / dx) + 1, round(Int, ly / dx) + 1
    g = Grid(nx, ny, lx, ly)
    x = range(0, lx; length = nx)
    zs = [6 * (sqrt(xx + 5e3) - sqrt(5e3)) + 1 for xx in x, _ in 1:ny]
    zb = zeros(nx, ny)
    mask = fill(Float64(GROUNDED), nx, ny); mask[1, :] .= LAND
    p = ModelParameters(; rho_w = 1000.0, rho_i = 910.0, g = 9.8, L = 334e3, cw = 4220.0, ct = 7.5e-8, br = 0.1, lr = 2.0,
                        e_v = 0.0, mdot_includes_G = false, mdot_includes_frictional = false, mdot_includes_qT = false)
    sl = RegularizedCoulombSlidingLaw(0.25)
    st = State(g)
    set_initial_conditions!(st, g, p, sl, mask, fill(2.5e-25, nx, ny), zb, zs, fill(0.01, nx, ny), zeros(nx, ny),
                            fill(1e-6, nx + 1, ny), zeros(nx, ny + 1), fill(5.79e-9, nx, ny), zeros(nx + 1, ny), zeros(nx, ny + 1))
    sim = Simulation(g, st, nsteps, floattype(dt), p, ConstantMeltInput(), sl;
                     ps = PicardSolver(500, floattype(1e-6), make_ls(g), g), (sheet == :spd ? SPD_SHEET : (;))...)
    run!(sim)
    s = sim.state
    return (; N = Array(s.N), b = Array(s.b), h = Array(s.h), grounded = Array(s.mask) .== GROUNDED,
            unconverged = sim.wb.unconverged[])
end

rel2(a, b, m) = norm(a[m] .- b[m]) / norm(b[m])

# Final N, b and h of run `r` against the LU reference `ref` on the grounded cells, relative L2 difference below `rtol`.
function test_against(r, ref, rtol; label = "")
    @test r.unconverged == 0
    for f in (:N, :b, :h)
        d = rel2(getfield(r, f), getfield(ref, f), ref.grounded)
        println("  $label $f: relative L2 difference to LU ", round(d; sigdigits = 2))
        @test d < rtol
    end
end
