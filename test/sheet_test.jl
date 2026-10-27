@testset "Sheet presets and typed options (ARCH-10)" begin

    nx, ny = 6, 6
    grid = Grid(nx, ny, 1e3, 1e3)
    sl = RegularizedCoulombSlidingLaw(0.25)
    mask = fill(GROUNDED, nx, ny); mask[end, :] .= OCEAN; mask[1, :] .= LAND; mask[:, 1] .= OTHER_BASIN
    zb = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
    dt = floattype(10800.0)

    function make_state(p)
        state = State(grid)
        set_initial_conditions!(state, grid, p, sl, mask, fill(5e-25, nx, ny), zb, zb .+ 500.0, fill(0.01, nx, ny), fill(0.06, nx, ny),
                                fill(1e-6, nx + 1, ny), zeros(nx, ny + 1), fill(1e-8, nx, ny), zeros(nx + 1, ny), zeros(nx, ny + 1))
        return state
    end
    lu_ps() = PicardSolver(500, floattype(1e-9), LUDirectSolver(grid), grid)
    chol_ps() = PicardSolver(500, floattype(1e-9), CholeskyDirectSolver(grid), grid)
    options(sim) = map(x -> typeof(x), (sim.cf, sim.up, sim.hp, sim.gcp, sim.kfs, sim.observer))

    p = ModelParameters(e_v = 0.0, b_min = 1e-6)

    @testset "GlaDS2Sheet is the default, in the short and the long form" begin
        short = Simulation(grid, make_state(p), 1, dt, p, ConstantMeltInput(), sl; ps = lu_ps())
        long  = Simulation(grid, make_state(p), 1, dt, p, ConstantMeltInput(), sl; ps = lu_ps(), sheet = GlaDS2Sheet())
        @test options(short) == options(long) ==
              (UnfilledCavities, IceUplift, SheetTopPotential, ConsistentGapCoupling, ImplicitUpwind, NoObserver)
        step!(short); step!(long)
        @test Array(short.state.h) == Array(long.state.h)
    end

    @testset "ShaktiSheet(original = true) is the old keyword set, step for step" begin
        preset = Simulation(grid, make_state(p), 2, dt, p, ConstantMeltInput(), sl; ps = chol_ps(), sheet = ShaktiSheet(original = true))
        old    = Simulation(grid, make_state(p), 2, dt, p, ConstantMeltInput(), sl; ps = chol_ps(),
                            cavity_filling = FilledCavities(), uplift = NoUplift(), hydraulic_potential = BedPotential(),
                            gap_coupling = ExplicitGapCoupling())
        @test options(preset) == options(old) ==
              (FilledCavities, NoUplift, BedPotential, ExplicitGapCoupling, Arithmetic, NoObserver)
        run!(preset); run!(old)
        @test Array(preset.state.h) == Array(old.state.h) && Array(preset.state.b) == Array(old.state.b)
        @test typeof(ShaktiSheet().hp) == SheetTopPotential && typeof(ShaktiSheet().gcp) == ConsistentGapCoupling
    end

    @testset "a keyword overrides the sheet's field; k_face_choice takes the scheme itself" begin
        sim = Simulation(grid, make_state(p), 1, dt, p, ConstantMeltInput(), sl; ps = chol_ps(), sheet = ShaktiSheet(),
                         hydraulic_potential = BedPotential(), k_face_choice = Harmonic())
        @test options(sim)[1:5] == (FilledCavities, NoUplift, BedPotential, ConsistentGapCoupling, Harmonic)
        @test_throws ErrorException Simulation(grid, make_state(p), 1, dt, p, ConstantMeltInput(), sl; ps = lu_ps(), k_face_choice = Arithmetic())
        @test_throws ErrorException Simulation(grid, make_state(p), 1, dt, p, ConstantMeltInput(), sl; ps = lu_ps(), k_face_choice = "averaged")
    end

    @testset "an observer passed as an object" begin
        sim = Simulation(grid, make_state(p), 2, dt, p, ConstantMeltInput(), sl; ps = lu_ps(), observer = LiveObserver(["h", "b_empty"], 0:2))
        run!(sim)
        @test size(sim.observer.history["h"], 3) == 3 && sim.observer.history["h"][:, :, end] == Array(sim.state.h)
        @test_throws ErrorException Simulation(grid, make_state(p), 1, dt, p, ConstantMeltInput(), sl; ps = lu_ps(), observer = LiveObserver(["nope"], 0:1))
    end

    @testset "OPTION_RULES: one table of the refused and warned combinations" begin
        @test all(r -> r[2] in (:error, :warn, :warn_once), Shakti.OPTION_RULES)
        o(; kw...) = (; cf = UnfilledCavities(), up = IceUplift(), gcp = ConsistentGapCoupling(), free_surface = true, kfs = ImplicitUpwind(), k_face_default = true,
                       ps = lu_ps(), ds = NoDiffusion(), cnc = NoCellNClamping(), p, gs = FullyImplicitGapScheme(), kw...)
        @test Shakti.check_options(o()) === nothing                                         # the default sheet passes every row
        @test_throws r"does not conserve water" Shakti.check_options(o(kfs = Arithmetic(), k_face_default = false))
        @test_throws r"never refills" Shakti.check_options(o(kfs = Harmonic(), k_face_default = false))
        @test_throws r"non-symmetric head matrix: it needs" Shakti.check_options(o(ps = chol_ps()))
        @test_throws r"tighter than \[0, Inf\)" Shakti.check_options(o(p = ModelParameters(e_v = 0.0, N_min = 1e3)))
        @test_logs (:warn, r"lags the water depth") Shakti.check_options(o(kfs = Upwind(), k_face_default = false))
        @test_logs Shakti.check_options(o(cf = FilledCavities(), up = NoUplift(), free_surface = false, kfs = Arithmetic(), ps = chol_ps(),
                                          p = ModelParameters(e_v = 0.0, N_min = 1e3)))   # the filled sheet takes averaged faces and N clamps
    end

    @testset "C1: a finite gap runaway stops the run; implicit_upwind on the dry floor with ExplicitGapScheme warns" begin
        # The rule warns once per session, so its predicate and message are checked rather than the log.
        dry = (; cf = FilledCavities(), up = NoUplift(), gcp = ConsistentGapCoupling(), free_surface = false, kfs = ImplicitUpwind(),
                 k_face_default = false, ps = lu_ps(), ds = NoDiffusion(), cnc = NoCellNClamping(), p, gs = ExplicitGapScheme())
        fires(o) = [r[3](o) for r in Shakti.OPTION_RULES if r[2] === :warn_once && r[1](o)]
        @test any(m -> occursin("can run away", m), fires(dry))
        @test !any(m -> occursin("can run away", m), fires(merge(dry, (; gs = FullyImplicitGapScheme()))))
        @test !any(m -> occursin("can run away", m), fires(merge(dry, (; gcp = ExplicitGapCoupling()))))

        # The stuck-cells fixture: nearly rigid ice, a 1 um gap and a point source. The explicit gap update reaches
        # b = 2 km (H = 500 m) in step 3; before the check it ran on to 3e11 m in step 4.
        stuck = State(grid)
        ieb = zeros(nx, ny); ieb[3, 3] = 3 / (grid.dx * grid.dy)
        set_initial_conditions!(stuck, grid, p, sl, mask, fill(1e-30, nx, ny), zb, zb .+ 500.0, fill(1e-6, nx, ny), fill(0.06, nx, ny),
                                fill(3e-7, nx + 1, ny), zeros(nx, ny + 1), ieb, zeros(nx + 1, ny), zeros(nx, ny + 1))
        sim = Simulation(grid, stuck, 4, dt, p, ConstantMeltInput(), sl; gap_scheme = ExplicitGapScheme(), cavity_filling = FilledCavities(),
                         uplift = NoUplift(), k_face_choice = "implicit_upwind", hydraulic_potential = BedPotential(),
                         ps = PicardSolver(500, floattype(1e-12), LUDirectSolver(grid), grid))
        @test_throws r"gap runaway" Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            for _ in 1:4
                step!(sim)
            end
        end
        # The fixture's first steps end unconverged (the filled sheet without storage takes up the moulin only by creep
        # opening at N < 0): each one is counted in the water budget (review C2).
        @test sim.wb.unconverged[] >= 1
        @test water_created(sim).unconverged == sim.wb.unconverged[]
    end
end
