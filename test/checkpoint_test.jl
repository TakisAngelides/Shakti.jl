    @testset "Checkpoint / Restart" begin

        # Same nontrivial mask/state as the solver testsets above, run through
        # a real elliptic-head-scheme Simulation (not just isolated solver
        # calls) since checkpoint/restart is a run!-level concern.
        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        p = ModelParameters(e_v = 0.0)
        mi = ConstantMeltInput()
        sl = RegularizedCoulombSlidingLaw(0.25)

        mask = fill(GROUNDED, nx, ny)
        mask[end, :] .= OCEAN
        mask[1, :]   .= LAND
        mask[:, 1]   .= OTHER_BASIN

        A_visc = fill(5e-25, nx, ny)
        zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
        zs     = zb .+ 500.0
        b      = fill(0.01, nx, ny)
        G      = fill(0.06, nx, ny)
        ub_x   = fill(1e-6, nx + 1, ny)
        ub_y   = zeros(nx, ny + 1)
        ieb    = zeros(nx, ny)
        ieb[3, 3] = 3 / (grid.dx * grid.dy)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)

        dt = 3600.0 # 1h, same scale as test/reproduce_section_3_3.jl
        tsteps = 8
        tracked_obs = ["h", "b"]
        tracked_times = 0:tsteps

        function make_sim(sim_tsteps, which_file_writer, path)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ls = CholeskyDirectSolver(grid)
            ps = PicardSolver(500, 1e-6, ls, grid)
            return Simulation(grid, state, sim_tsteps, floattype(dt), p, "fully_implicit", tracked_obs, mi, sl;
                               ps = ps, which_observer = "IO", which_file_writer = which_file_writer,
                               tracked_times = tracked_times, path = path)
        end

        mktempdir() do dir

            @testset "NetCDF" begin
                full_path = joinpath(dir, "full.nc")
                sim_full = make_sim(tsteps, "NetCDF", full_path)
                run!(sim_full)

                # "Crashed" run: only gets to t=5 (as if killed there), but
                # checkpoints every 3 steps -- so its last checkpoint is at
                # t=3, while the observer already durably wrote t=4 and t=5
                # to crash_path before the "crash".
                crash_path = joinpath(dir, "crash.nc")
                checkpoint_path = joinpath(dir, "ckpt.jld2")
                sim_crash = make_sim(5, "NetCDF", crash_path)
                run!(sim_crash; checkpoint_every = 3, checkpoint_path = checkpoint_path)

                # Resume into a brand-new Simulation/State (nothing shared in
                # memory with sim_crash -- the checkpoint file is the only
                # link), back to the original tsteps=8. This replays t=4,5
                # (already in crash_path) before reaching new ground at 6,7,8.
                sim_resumed = make_sim(tsteps, "NetCDF", crash_path)
                run!(sim_resumed; restart_path = checkpoint_path)

                @test Array(sim_resumed.state.h) ≈ Array(sim_full.state.h)
                @test Array(sim_resumed.state.b) ≈ Array(sim_full.state.b)
                @test sim_resumed.total_time[] ≈ sim_full.total_time[]

                h_full    = NetCDF.ncread(full_path, "h")
                h_resumed = NetCDF.ncread(crash_path, "h")
                @test size(h_resumed) == size(h_full) # catches any duplicated/dropped time slice
                @test h_resumed ≈ h_full
            end

            @testset "HDF5" begin
                full_path = joinpath(dir, "full.h5")
                sim_full = make_sim(tsteps, "HDF5", full_path)
                run!(sim_full)

                crash_path = joinpath(dir, "crash.h5")
                checkpoint_path = joinpath(dir, "ckpt2.jld2")
                sim_crash = make_sim(5, "HDF5", crash_path)
                run!(sim_crash; checkpoint_every = 3, checkpoint_path = checkpoint_path)

                sim_resumed = make_sim(tsteps, "HDF5", crash_path)
                run!(sim_resumed; restart_path = checkpoint_path)

                @test Array(sim_resumed.state.h) ≈ Array(sim_full.state.h)
                @test Array(sim_resumed.state.b) ≈ Array(sim_full.state.b)
                @test sim_resumed.total_time[] ≈ sim_full.total_time[]

                HDF5.h5open(full_path, "r") do ffull
                    HDF5.h5open(crash_path, "r") do fresumed
                        @test size(fresumed["h"]) == size(ffull["h"])
                        @test read(fresumed["h"]) ≈ read(ffull["h"])
                    end
                end
            end

            @testset "JLD2" begin
                full_path = joinpath(dir, "full.jld2")
                sim_full = make_sim(tsteps, "JLD2", full_path)
                run!(sim_full)

                crash_path = joinpath(dir, "crash.jld2")
                checkpoint_path = joinpath(dir, "ckpt3.jld2")
                sim_crash = make_sim(5, "JLD2", crash_path)
                run!(sim_crash; checkpoint_every = 3, checkpoint_path = checkpoint_path)

                sim_resumed = make_sim(tsteps, "JLD2", crash_path)
                run!(sim_resumed; restart_path = checkpoint_path)

                @test Array(sim_resumed.state.h) ≈ Array(sim_full.state.h)
                @test Array(sim_resumed.state.b) ≈ Array(sim_full.state.b)
                @test sim_resumed.total_time[] ≈ sim_full.total_time[]

                JLD2.jldopen(full_path, "r") do ffull
                    JLD2.jldopen(crash_path, "r") do fresumed
                        for (idx, t) in enumerate(tracked_times)
                            @test fresumed["h/$idx"] ≈ ffull["h/$idx"]
                            @test fresumed["b/$idx"] ≈ ffull["b/$idx"]
                        end
                    end
                end
            end

            @testset "CSV" begin
                full_path = joinpath(dir, "full.csv")
                sim_full = make_sim(tsteps, "CSV", full_path)
                run!(sim_full)

                crash_path = joinpath(dir, "crash.csv")
                checkpoint_path = joinpath(dir, "ckpt4.jld2")
                sim_crash = make_sim(5, "CSV", crash_path)
                run!(sim_crash; checkpoint_every = 3, checkpoint_path = checkpoint_path)

                sim_resumed = make_sim(tsteps, "CSV", crash_path)
                run!(sim_resumed; restart_path = checkpoint_path)

                @test Array(sim_resumed.state.h) ≈ Array(sim_full.state.h)
                @test Array(sim_resumed.state.b) ≈ Array(sim_full.state.b)
                @test sim_resumed.total_time[] ≈ sim_full.total_time[]

                rows_full    = collect(CSV.File(full_path))
                rows_resumed = collect(CSV.File(crash_path))
                @test length(rows_resumed) == length(tracked_times) # catches duplicated rows from replaying t=4,5
                @test length(rows_resumed) == length(rows_full)
                for (row_full, row_resumed) in zip(rows_full, rows_resumed)
                    @test row_resumed.t == row_full.t
                    @test row_resumed.h_mean ≈ row_full.h_mean
                    @test row_resumed.b_mean ≈ row_full.b_mean
                end
            end

        end

    end

    @testset "Checkpoint / Extend" begin

        # Same setup as "Checkpoint / Restart" above, but exercising extend_path instead:
        # a run that reaches ITS OWN target cleanly (not killed partway), checkpointed only via
        # the always-checkpoint-the-last-step behavior (checkpoint_every is set far past its own
        # tsteps, so the periodic schedule never fires), then continued into a NEW, freshly
        # tracked-times file for a further stretch of steps.
        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        p = ModelParameters(e_v = 0.0)
        mi = ConstantMeltInput()
        sl = RegularizedCoulombSlidingLaw(0.25)

        mask = fill(GROUNDED, nx, ny)
        mask[end, :] .= OCEAN
        mask[1, :]   .= LAND
        mask[:, 1]   .= OTHER_BASIN

        A_visc = fill(5e-25, nx, ny)
        zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
        zs     = zb .+ 500.0
        b      = fill(0.01, nx, ny)
        G      = fill(0.06, nx, ny)
        ub_x   = fill(1e-6, nx + 1, ny)
        ub_y   = zeros(nx, ny + 1)
        ieb    = zeros(nx, ny)
        ieb[3, 3] = 3 / (grid.dx * grid.dy)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)

        dt = 3600.0        # 1h, same scale as the Restart testset above
        n_first, n_more = 5, 3 # split of the same 8-step reference run used for comparison
        tracked_obs = ["h", "b"]

        # Unlike the Restart testset's make_sim, tracked_times is sized to EACH call's own
        # sim_tsteps (0:sim_tsteps) rather than a single fixed range shared across phases --
        # extend_path always starts a fresh file sized for "how many more steps", not the
        # original run's total (see run.jl's extend_path docstring).
        function make_sim(sim_tsteps, which_file_writer, path)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ls = CholeskyDirectSolver(grid)
            ps = PicardSolver(500, 1e-6, ls, grid)
            return Simulation(grid, state, sim_tsteps, floattype(dt), p, "fully_implicit", tracked_obs, mi, sl;
                               ps = ps, which_observer = "IO", which_file_writer = which_file_writer,
                               tracked_times = 0:sim_tsteps, path = path)
        end

        mktempdir() do dir

            @testset "NetCDF" begin
                full_path = joinpath(dir, "full.nc")
                sim_full = make_sim(n_first + n_more, "NetCDF", full_path)
                run!(sim_full)

                first_path = joinpath(dir, "first.nc")
                checkpoint_path = joinpath(dir, "ckpt.jld2")
                sim_first = make_sim(n_first, "NetCDF", first_path)
                run!(sim_first; checkpoint_every = 1000, checkpoint_path = checkpoint_path)

                extended_path = joinpath(dir, "extended.nc")
                sim_extended = make_sim(n_more, "NetCDF", extended_path)
                run!(sim_extended; extend_path = checkpoint_path)

                @test Array(sim_extended.state.h) ≈ Array(sim_full.state.h)
                @test Array(sim_extended.state.b) ≈ Array(sim_full.state.b)
                @test sim_extended.total_time[] ≈ sim_full.total_time[]

                time_extended = NetCDF.ncread(extended_path, "time")
                @test time_extended[1] ≈ n_first * dt   # frame 0 stamped with the loaded total_time, not 0
                @test time_extended[end] ≈ sim_full.total_time[]

                h_full     = NetCDF.ncread(full_path, "h")
                h_extended = NetCDF.ncread(extended_path, "h")
                @test h_extended[:, :, end] ≈ h_full[:, :, end] # final frame of the new file matches the continuous reference run's final frame
            end

            @testset "HDF5" begin
                full_path = joinpath(dir, "full.h5")
                sim_full = make_sim(n_first + n_more, "HDF5", full_path)
                run!(sim_full)

                first_path = joinpath(dir, "first.h5")
                checkpoint_path = joinpath(dir, "ckpt2.jld2")
                sim_first = make_sim(n_first, "HDF5", first_path)
                run!(sim_first; checkpoint_every = 1000, checkpoint_path = checkpoint_path)

                extended_path = joinpath(dir, "extended.h5")
                sim_extended = make_sim(n_more, "HDF5", extended_path)
                run!(sim_extended; extend_path = checkpoint_path)

                @test Array(sim_extended.state.h) ≈ Array(sim_full.state.h)
                @test Array(sim_extended.state.b) ≈ Array(sim_full.state.b)
                @test sim_extended.total_time[] ≈ sim_full.total_time[]

                HDF5.h5open(full_path, "r") do ffull
                    HDF5.h5open(extended_path, "r") do fext
                        time_ext = read(fext["time"])
                        @test time_ext[1] ≈ n_first * dt
                        @test time_ext[end] ≈ sim_full.total_time[]
                        @test read(fext["h"])[:, :, end] ≈ read(ffull["h"])[:, :, end]
                    end
                end
            end

            @testset "JLD2" begin
                full_path = joinpath(dir, "full.jld2")
                sim_full = make_sim(n_first + n_more, "JLD2", full_path)
                run!(sim_full)

                first_path = joinpath(dir, "first.jld2")
                checkpoint_path = joinpath(dir, "ckpt3.jld2")
                sim_first = make_sim(n_first, "JLD2", first_path)
                run!(sim_first; checkpoint_every = 1000, checkpoint_path = checkpoint_path)

                extended_path = joinpath(dir, "extended.jld2")
                sim_extended = make_sim(n_more, "JLD2", extended_path)
                run!(sim_extended; extend_path = checkpoint_path)

                @test Array(sim_extended.state.h) ≈ Array(sim_full.state.h)
                @test Array(sim_extended.state.b) ≈ Array(sim_full.state.b)
                @test sim_extended.total_time[] ≈ sim_full.total_time[]

                JLD2.jldopen(full_path, "r") do ffull
                    JLD2.jldopen(extended_path, "r") do fext
                        @test fext["total_time/1"] ≈ n_first * dt   # frame 0 (idx 1) stamped with the loaded total_time, not 0
                        @test fext["total_time/$(n_more + 1)"] ≈ sim_full.total_time[]
                        @test fext["h/$(n_more + 1)"] ≈ ffull["h/$(n_first + n_more + 1)"]
                    end
                end
            end

            @testset "CSV" begin
                full_path = joinpath(dir, "full.csv")
                sim_full = make_sim(n_first + n_more, "CSV", full_path)
                run!(sim_full)

                first_path = joinpath(dir, "first.csv")
                checkpoint_path = joinpath(dir, "ckpt4.jld2")
                sim_first = make_sim(n_first, "CSV", first_path)
                run!(sim_first; checkpoint_every = 1000, checkpoint_path = checkpoint_path)

                extended_path = joinpath(dir, "extended.csv")
                sim_extended = make_sim(n_more, "CSV", extended_path)
                run!(sim_extended; extend_path = checkpoint_path)

                @test Array(sim_extended.state.h) ≈ Array(sim_full.state.h)
                @test Array(sim_extended.state.b) ≈ Array(sim_full.state.b)
                @test sim_extended.total_time[] ≈ sim_full.total_time[]

                rows_full = collect(CSV.File(full_path))
                rows_ext  = collect(CSV.File(extended_path))
                @test length(rows_ext) == n_more + 1
                @test rows_ext[1].total_time ≈ n_first * dt
                @test rows_ext[end].h_mean ≈ rows_full[end].h_mean
                @test rows_ext[end].b_mean ≈ rows_full[end].b_mean
            end

        end

    end

    @testset "Checkpoint / Extend errors" begin
        nx, ny = 4, 4
        grid = Grid(nx, ny, 1e3, 1e3)
        p = ModelParameters(e_v = 0.0)
        mi = ConstantMeltInput()
        sl = RegularizedCoulombSlidingLaw(0.25)
        mask = fill(GROUNDED, nx, ny)
        state = State(grid)
        set_initial_conditions!(state, grid, p, sl, mask, fill(5e-25, nx, ny), zeros(nx, ny), fill(500.0, nx, ny),
                                 fill(0.01, nx, ny), fill(0.06, nx, ny), fill(1e-6, nx + 1, ny), zeros(nx, ny + 1),
                                 zeros(nx, ny), zeros(nx + 1, ny), zeros(nx, ny + 1))
        ls = CholeskyDirectSolver(grid)
        ps = PicardSolver(500, 1e-6, ls, grid)
        sim = Simulation(grid, state, 3, floattype(3600.0), p, "fully_implicit", String[], mi, sl; ps = ps)

        mktempdir() do dir
            mkpath(dir)
            @test_throws ErrorException run!(sim; restart_path = joinpath(dir, "a.jld2"), extend_path = joinpath(dir, "b.jld2"))
        end
    end
