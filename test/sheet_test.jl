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
        long  = Simulation(grid, make_state(p), 1, dt, p, String[], ConstantMeltInput(), sl; ps = lu_ps(), sheet = GlaDS2Sheet())
        @test options(short) == options(long) ==
              (UnfilledCavities, IceUplift, SheetTopPotential, ConsistentGapCoupling, ImplicitUpwind, NoObserver)
        step!(short); step!(long)
        @test Array(short.state.h) == Array(long.state.h)
    end

    @testset "ShaktiSheet(original = true) is the old keyword set, step for step" begin
        preset = Simulation(grid, make_state(p), 2, dt, p, ConstantMeltInput(), sl; ps = chol_ps(), sheet = ShaktiSheet(original = true))
        old    = Simulation(grid, make_state(p), 2, dt, p, "fully_implicit", String[], ConstantMeltInput(), sl; ps = chol_ps(),
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
        @test_throws ErrorException Simulation(grid, make_state(p), 1, dt, p, ["h"], ConstantMeltInput(), sl; ps = lu_ps(),
                                               which_observer = "Live", tracked_times = 0:1, observer = LiveObserver(["h"], 0:1))
        @test_throws ErrorException Simulation(grid, make_state(p), 1, dt, p, ConstantMeltInput(), sl; ps = lu_ps(), observer = LiveObserver(["nope"], 0:1))
    end
end
