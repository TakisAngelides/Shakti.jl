    @testset "CreepCutoff (SUHMO Eq. 7)" begin

        # --- Pure math: creep_length / creep_length_slope -----------------------------------

        @testset "creep_length / creep_length_slope" begin
            b_c = 0.05

            # StandardCreep: l_c = b unconditionally, dl_c/db = 1
            for b in (0.0, 0.01, 0.05, 0.2, 5.0)
                @test Shakti.creep_length(StandardCreep(), b, b_c) == b
                @test Shakti.creep_length_slope(StandardCreep(), b, b_c) == 1.0
            end

            # CreepCutoff: continuous at b = b_c (both branches must agree there)
            @test Shakti.creep_length(CreepCutoff(), b_c, b_c) ≈ b_c

            # Below cutoff: l_c = b^2/b_c
            @test Shakti.creep_length(CreepCutoff(), 0.02, b_c) ≈ 0.02^2 / b_c
            @test Shakti.creep_length_slope(CreepCutoff(), 0.02, b_c) ≈ 2 * 0.02 / b_c

            # Above cutoff: l_c = b, same as StandardCreep
            @test Shakti.creep_length(CreepCutoff(), 0.2, b_c) == 0.2
            @test Shakti.creep_length_slope(CreepCutoff(), 0.2, b_c) == 1.0

            # creep_length_slope matches a central finite difference of creep_length itself,
            # on both sides of the cutoff -- catches a slope/formula mismatch directly.
            for b in (0.01, 0.04, 0.06, 0.3)
                eps = 1e-6
                fd = (Shakti.creep_length(CreepCutoff(), b + eps, b_c) -
                      Shakti.creep_length(CreepCutoff(), b - eps, b_c)) / (2 * eps)
                @test Shakti.creep_length_slope(CreepCutoff(), b, b_c) ≈ fd atol = 1e-4
            end
        end

        # --- implicit_creep_update: closed-form solve of b_new = b_old + dt*(opening - C*l_c(b_new)) --

        @testset "implicit_creep_update solves its own defining equation" begin
            residual(cls, b_old, opening, C, dt, b_c, b_new) =
                b_new - b_old - dt * (opening - C * Shakti.creep_length(cls, b_new, b_c))

            b_c = 0.05
            cases = [
                (b_old = 0.001, opening = 0.0,  C = 10.0,  dt = 3600.0), # strong closure, no opening -- must land near 0 (<= b_c)
                (b_old = 0.001, opening = 1e-3, C = 1e-12, dt = 3600.0), # opening dominates, negligible closure -- must land well above b_c
                (b_old = 0.05,  opening = 1e-6, C = 1e-6,  dt = 3600.0), # starts exactly at b_c
                (b_old = 0.01,  opening = 1e-6, C = 0.0,   dt = 3600.0), # C=0 -- exercises the a=0 linear fallback inside the CreepCutoff branch
            ]

            for c in cases
                b_new = Shakti.implicit_creep_update(CreepCutoff(), c.b_old, c.opening, c.C, c.dt, b_c)
                @test isfinite(b_new)
                @test b_new >= 0
                @test abs(residual(CreepCutoff(), c.b_old, c.opening, c.C, c.dt, b_c, b_new)) < 1e-8

                # StandardCreep's own (unconditionally linear) formula must also solve ITS OWN equation
                b_new_std = Shakti.implicit_creep_update(StandardCreep(), c.b_old, c.opening, c.C, c.dt, b_c)
                @test abs(residual(StandardCreep(), c.b_old, c.opening, c.C, c.dt, b_c, b_new_std)) < 1e-8
            end

            # Explicit branch coverage: the two extreme cases above must land on opposite sides of b_c.
            b_below = Shakti.implicit_creep_update(CreepCutoff(), 0.001, 0.0, 10.0, 3600.0, b_c)
            @test b_below <= b_c
            b_above = Shakti.implicit_creep_update(CreepCutoff(), 0.001, 1e-3, 1e-12, 3600.0, b_c)
            @test b_above > b_c
        end

        # --- fully_implicit_creep_update: closed-form solve with beta AND l_c both piecewise ---

        @testset "fully_implicit_creep_update solves its own defining equation, either br/b_c ordering" begin
            residual(cls, opening0, gamma, C, dt, br, b_c, b_new) =
                b_new - opening0 - dt * gamma * max(zero(b_new), br - b_new) + dt * C * Shakti.creep_length(cls, b_new, b_c)

            dt = 3600.0

            # Wide-window cases, each engineered so the closure/opening terms only perturb the
            # equilibrium by a small fraction of the gap to the nearest threshold -- so the branch
            # each case lands in is unambiguous without needing to solve it by hand first.
            branch_cases = [
                (label = "(i) beta active, l_c cutoff",   opening0 = 0.1, gamma = 1e-6, C = 1e-6, br = 1.0,  b_c = 0.3),
                (label = "(ii) beta active, l_c linear",  opening0 = 0.3, gamma = 1e-6, C = 1e-6, br = 1.0,  b_c = 0.01),
                (label = "(iii) beta zero, l_c cutoff",   opening0 = 0.2, gamma = 1e-6, C = 1e-6, br = 0.01, b_c = 0.5),
                (label = "(iv) beta zero, l_c linear",    opening0 = 0.2, gamma = 1e-6, C = 1e-6, br = 0.01, b_c = 0.05),
            ]

            for c in branch_cases
                b_new = Shakti.fully_implicit_creep_update(CreepCutoff(), c.opening0, 0.0, c.gamma, c.C, dt, c.br, c.b_c)
                @test isfinite(b_new)
                @test b_new >= 0
                @test abs(residual(CreepCutoff(), c.opening0, c.gamma, c.C, dt, c.br, c.b_c, b_new)) < 1e-8
            end

            # Explicit branch-membership check, confirming genuine coverage of all four branches
            # (not just that the residual happens to be small everywhere).
            b1 = Shakti.fully_implicit_creep_update(CreepCutoff(), 0.1, 0.0, 1e-6, 1e-6, dt, 1.0, 0.3)
            @test b1 < 1.0 && b1 <= 0.3   # branch (i)
            b2 = Shakti.fully_implicit_creep_update(CreepCutoff(), 0.3, 0.0, 1e-6, 1e-6, dt, 1.0, 0.01)
            @test b2 < 1.0 && b2 > 0.01   # branch (ii)
            b3 = Shakti.fully_implicit_creep_update(CreepCutoff(), 0.2, 0.0, 1e-6, 1e-6, dt, 0.01, 0.5)
            @test b3 >= 0.01 && b3 <= 0.5 # branch (iii) -- only reachable when br <= b_c
            b4 = Shakti.fully_implicit_creep_update(CreepCutoff(), 0.2, 0.0, 1e-6, 1e-6, dt, 0.01, 0.05)
            @test b4 >= 0.01 && b4 > 0.05 # branch (iv)

            # Broader residual-only sweep (no branch prediction needed) across both br/b_c
            # orderings, StandardCreep included -- robustness check beyond the hand-picked cases.
            sweep_cases = [
                (opening0 = 0.02, gamma = 5e-7, C = 1e-4, br = 0.1,  b_c = 0.03), # br > b_c
                (opening0 = 0.15, gamma = 3e-7, C = 1e-5, br = 0.1,  b_c = 0.03),
                (opening0 = 0.02, gamma = 5e-7, C = 1e-4, br = 0.03, b_c = 0.1),  # br < b_c
                (opening0 = 0.15, gamma = 3e-7, C = 1e-5, br = 0.03, b_c = 0.1),
                (opening0 = 0.05, gamma = 0.0,  C = 1e-4, br = 0.1,  b_c = 0.1),  # br == b_c, gamma == 0
            ]
            for c in sweep_cases, cls in (StandardCreep(), CreepCutoff())
                b_new = Shakti.fully_implicit_creep_update(cls, c.opening0, 0.0, c.gamma, c.C, dt, c.br, c.b_c)
                @test isfinite(b_new)
                @test b_new >= 0
                @test abs(residual(cls, c.opening0, c.gamma, c.C, dt, c.br, c.b_c, b_new)) < 1e-8
            end

            # The kink (review C3): starting gaps whose new gap is exactly b_c (< br). Rounding put branch (i)'s root
            # 4e-14 above b_c and branch (ii)'s 2e-17 below it, and branch (iv) was returned unchecked, 4e-6 m low.
            m, g, C, dtk, br, b_c = 6.396420948918853e-10, 1.1100887267871447e-7, 2.6946008753917766e-8, 711.7103506433916, 0.1, 0.05
            b0s = b_c * (1 + dtk * (g + C)) - dtk * (m + g * br)
            for b0 in (prevfloat(b0s), b0s, nextfloat(b0s), 0.04999655333958234)
                b_new = Shakti.fully_implicit_creep_update(CreepCutoff(), b0, m, g, C, dtk, br, b_c)
                @test abs(residual(CreepCutoff(), b0 + dtk * m, g, C, dtk, br, b_c, b_new)) < 1e-12
                @test abs(b_new - b_c) < 1e-12
            end
        end

        # --- End-to-end: CreepCutoff through the real kernel launches, via a Simulation --------

        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        sl = RegularizedCoulombSlidingLaw(0.25)

        mask = fill(GROUNDED, nx, ny)
        mask[end, :] .= OCEAN
        mask[1, :]   .= LAND
        mask[:, 1]   .= OTHER_BASIN

        A_visc = fill(5e-25, nx, ny)
        zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
        zs     = zb .+ 500.0
        b0     = fill(0.01, nx, ny)
        G      = fill(0.06, nx, ny)
        ub_x   = fill(1e-6, nx + 1, ny)
        ub_y   = zeros(nx, ny + 1)
        ieb    = zeros(nx, ny)
        ieb[3, 3] = 3 / (grid.dx * grid.dy)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)

        @testset "Simulation construction picks CreepCutoff/StandardCreep from p.b_c" begin
            p_cutoff = ModelParameters(e_v = 0.0, b_c = 0.05)
            state_cutoff = State(grid)
            set_initial_conditions!(state_cutoff, grid, p_cutoff, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            @test state_cutoff.b[3, 3] == 0.01 # sanity: below p_cutoff.b_c=0.05, so the cutoff branch below is actually exercised
            @test state_cutoff.lc[3, 3] ≈ state_cutoff.b[3, 3]^2 / p_cutoff.b_c # l_c = b^2/b_c below the cutoff, computed through the real set_initial_conditions!/compute_lc! path

            ls = CholeskyDirectSolver(grid)
            ps = PicardSolver(50, floattype(1e-6), ls, grid)
            sim_cutoff = Simulation(grid, state_cutoff, 1, floattype(3600.0), p_cutoff, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps)
            @test sim_cutoff.cls isa CreepCutoff

            p_std = ModelParameters(e_v = 0.0) # b_c defaults to 0.0
            @test p_std.b_c == 0.0
            state_std = State(grid)
            set_initial_conditions!(state_std, grid, p_std, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            @test state_std.lc == state_std.b # StandardCreep: l_c = b everywhere

            ls2 = CholeskyDirectSolver(grid)
            ps2 = PicardSolver(50, floattype(1e-6), ls2, grid)
            sim_std = Simulation(grid, state_std, 1, floattype(3600.0), p_std, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps2)
            @test sim_std.cls isa StandardCreep
        end

        @testset "CreepCutoff combines with FullyImplicitGapScheme (no longer restricted)" begin
            p = ModelParameters(e_v = 0.0, b_c = 0.05)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)

            ls = CholeskyDirectSolver(grid)
            ps = PicardSolver(50, floattype(1e-6), ls, grid)
            sim = Simulation(grid, state, 3, floattype(3600.0), p, ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps)

            @test sim.cls isa CreepCutoff
            @test sim.gs isa FullyImplicitGapScheme

            for _ in 1:3
                step!(sim)
            end

            @test all(isfinite, Array(sim.state.b))
            @test all(>=(0), Array(sim.state.b))
            @test all(isfinite, Array(sim.state.h))
            @test all(isfinite, Array(sim.state.N))
        end

        @testset "gap_scheme keyword: FullyImplicitGapScheme by default, same run as the deprecated string form" begin
            p = ModelParameters(e_v = 0.0, b_c = 0.05)
            function build(args...; kw...)
                state = State(grid)
                set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
                ps = PicardSolver(50, floattype(1e-6), CholeskyDirectSolver(grid), grid)
                legacy = isempty(args) ? () : (args..., String[]) # the deprecated form: the gap scheme by name, then tracked_obs
                return Simulation(grid, state, 3, floattype(3600.0), p, legacy..., ConstantMeltInput(), sl; cavity_filling = FilledCavities(), uplift = NoUplift(), ps = ps, kw...)
            end
            sim_kw = build()
            sim_str = build("fully_implicit")
            @test sim_kw.gs isa FullyImplicitGapScheme
            @test build(; gap_scheme = ExplicitGapScheme()).gs isa ExplicitGapScheme
            @test build("explicit").gs isa ExplicitGapScheme
            @test_throws ErrorException build("crank_nicolson")
            for _ in 1:3
                step!(sim_kw)
                step!(sim_str)
            end
            @test Array(sim_kw.state.b) == Array(sim_str.state.b)
            @test Array(sim_kw.state.h) == Array(sim_str.state.h)
        end

    end


@testset "freeze_on_capacity_cell: freezing at C brings the gap to b_min" begin
    b_min, dt, br, lr = 1e-6, 3600.0, 0.1, 2.0
    for cls in (StandardCreep(), CreepCutoff()), (b0, ub, Cn) in ((0.05, 1e-6, 1e-9), (0.01, 3e-6, 1e-8), (0.2, 0.0, 1e-10))
        b_c = cls isa CreepCutoff ? 1e-7 : 0.0   # cutoff below b_min keeps l_c linear at the limit
        gamma = ub / lr
        Cap = Shakti.freeze_on_capacity_cell(b0, ub, Cn, dt, b_min, br, lr, cls, b_c, b_min, b_min)
        @test Cap > 0
        # fully implicit update with mdot/rho_i = -Cap lands exactly on b_min
        @test Shakti.fully_implicit_creep_update(cls, b0, -Cap, gamma, Cn, dt, br, b_c) ≈ b_min rtol = 1e-8
        # any less freezing leaves room above the floor
        @test Shakti.fully_implicit_creep_update(cls, b0, -0.9Cap, gamma, Cn, dt, br, b_c) > b_min
    end
    # a gap already at the floor with strong closure cannot freeze anything
    @test Shakti.freeze_on_capacity_cell(1e-6, 0.0, 1.0, 3600.0, 1e-6, 0.0, 2.0, StandardCreep(), 0.0, 1e-6, 1e-6) == 0
end

@testset "freeze_on_capacity_cell_diffusion: SUHMO local terms land on b_min (D = 0)" begin
    b_min, dt, br, lr, rho_i, n1 = 1e-6, 3600.0, 0.1, 2.0, 917.0, 2.0
    for (b0, ub, A, N, lc) in ((0.05, 1e-6, 1e-24, 2e5, 0.05), (0.01, 3e-6, 1e-24, 5e5, 0.01), (0.02, 1e-6, 1e-24, -1e5, 0.02))
        beta = max(0.0, (br - b0) / lr)
        Cn = A * abs(N)^n1 * N
        Cap = Shakti.freeze_on_capacity_cell_diffusion(b0, beta, ub, Cn, lc, dt, b_min, br, lr)
        @test Cap > 0
        diag, rhs = Shakti.b_diffusion_local_terms(b0, -rho_i * Cap, beta, ub, A, N, lc, rho_i, n1, dt, br, lr)
        @test rhs / (1 + diag) ≈ b_min rtol = 1e-8
    end
end

@testset "fully_implicit_creep_update: continuous and monotone at N < 0" begin
    # br = 0.1, dt*gamma = 1, dt*C = -0.5, m = 0: branch 1 is rejected for b0 >= ~0.05 and branch 2, the exponential
    # solution, then fell below br (b0 = 0.05 gave 0.0824 after 0.0933 at b0 = 0.04).
    br, b_c, dt = 0.1, 0.047, 1.0 # b_c = 0.05 puts a grid b0 exactly where CreepCutoff's solution is b_c, a rounding tie no branch accepts
    for cls in (Shakti.StandardCreep(), Shakti.CreepCutoff())
        for C in (-0.5, -0.2, 0.0, 0.3), gamma in (0.0, 1.0), m in (0.0, 0.01)
            G = [Shakti.fully_implicit_creep_update(cls, b0, m, gamma, C, dt, br, b_c) for b0 in 0.0:0.0025:0.3]
            @test all(diff(G) .>= -1e-14)
        end
        for b0 in (0.02, 0.05, 0.12), gamma in (0.0, 1.0)
            G = [Shakti.fully_implicit_creep_update(cls, b0, 0.0, gamma, C, dt, br, b_c) for C in -0.9:0.05:0.9]
            @test all(diff(G) .<= 1e-14) # more closure (larger N) never opens the gap
        end
    end
    # unchanged where C >= 0: branch 2 is already >= br whenever branch 1 is rejected
    for b0 in 0.0:0.01:0.3, gamma in (0.0, 1.0), C in (0.0, 0.3)
        b_below = Shakti.relax_update(b0, gamma * br, gamma + C, dt)
        @test Shakti.fully_implicit_creep_update(Shakti.StandardCreep(), b0, 0.0, gamma, C, dt, br, b_c) ==
              (b_below < br ? b_below : Shakti.relax_update(b0, 0.0, C, dt))
    end
end

@testset "CreepCutoff at N < 0: l_c = b under every gap scheme" begin
    cls, b_c = Shakti.CreepCutoff(), 0.05
    @test Shakti.creep_length(cls, 0.02, b_c, -1.0) == 0.02
    @test Shakti.creep_length_slope(cls, 0.02, b_c, -1.0) == 1.0
    @test Shakti.creep_length(cls, 0.02, b_c, 1.0) ≈ 0.02^2 / b_c
    # explicit (many small steps) and fully implicit (one step) agree on the growth of a gap below b_c at N < 0
    b0, m, C, T = 0.01, 1e-8, -1e-6, 1e4
    nsub = 10_000
    b = b0
    for _ in 1:nsub
        b += (T / nsub) * (m - C * Shakti.creep_length(cls, b, b_c, C))
    end
    b_impl = Shakti.fully_implicit_creep_update(cls, b0, m, 0.0, C, T, 1.0, b_c)
    @test b_impl ≈ b rtol = 1e-4
    @test b_impl > b0 * exp(-C * T) * 0.999 # linear (l_c = b) growth, not the quadratic one
end

@testset "row helpers keep Float32 (GPU, Metal has no Float64)" begin
    F = Float32
    z = zero(F)
    gu = Shakti.SchemeGapUpdate{Shakti.FullyImplicitGapScheme, Shakti.StandardCreep, F}(F(3600), F(0.1), F(2), F(0), false)
    r = @inferred Shakti.scheme_head_terms(gu, F(0.01), F(1e-6), F(Inf), F(1e-3), F(1e-3), F(1e-6), F(5e-25), F(1e6), F(100), F(0), F(4e6), z, F(1000), F(910), F(9.81), 2, z, F(0.01))
    @test r isa Tuple{F, F}
    s = @inferred Shakti.end_of_step_gap_slope(gu, F(0.01), z, F(1e-6), F(Inf), F(1e-3), F(1e-3), F(1e-6), F(5e-25), F(1e6), F(910), 2, F(1000), F(9.81))
    @test s isa F
end
