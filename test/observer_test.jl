    @testset "Observer / IO" begin

        nx, ny = 3, 4
        grid = Grid(nx, ny, 1e3, 1e3)
        state = State(grid)

        tracked_obs = ["h", "mdot"]
        tracked_times = [0, 2, 5]

        # h/mdot get a value that depends on t, so each tracked snapshot is
        # distinguishable -- this checks write2file!/observe! land the right
        # slice at the right index, not just any slice.
        fill_state!(state, t) = begin
            state.h    .= t .+ reshape(1:(nx * ny), nx, ny) ./ 10
            state.mdot .= t .- reshape(1:(nx * ny), nx, ny) ./ 10
        end

        mktempdir() do dir

            @testset "NetCDF" begin
                path = joinpath(dir, "out.nc")
                observer = IOObserver(tracked_obs, tracked_times, NetCDFFileWriter(), path)
                prepare!(observer, state)
                for t in 0:5
                    fill_state!(state, t)
                    observe!(observer, state, t, Float64(t))
                end
                finalize!(observer, state)

                h_nc    = NetCDF.ncread(path, "h")
                mdot_nc = NetCDF.ncread(path, "mdot")
                time_nc = NetCDF.ncread(path, "time")
                for (idx, t) in enumerate(tracked_times)
                    fill_state!(state, t)
                    @test h_nc[:, :, idx] ≈ Array(state.h)
                    @test mdot_nc[:, :, idx] ≈ Array(state.mdot)
                    @test time_nc[idx] ≈ Float64(t)
                end
            end

            @testset "HDF5" begin
                path = joinpath(dir, "out.h5")
                observer = IOObserver(tracked_obs, tracked_times, HDF5FileWriter(), path)
                prepare!(observer, state)
                for t in 0:5
                    fill_state!(state, t)
                    observe!(observer, state, t, Float64(t))
                end
                finalize!(observer, state)

                HDF5.h5open(path, "r") do file
                    for (idx, t) in enumerate(tracked_times)
                        fill_state!(state, t)
                        @test file["h"][:, :, idx] ≈ Array(state.h)
                        @test file["mdot"][:, :, idx] ≈ Array(state.mdot)
                        @test file["time"][idx] ≈ Float64(t)
                    end
                end
            end

            @testset "JLD2" begin
                path = joinpath(dir, "out.jld2")
                observer = IOObserver(tracked_obs, tracked_times, JLD2FileWriter(), path)
                prepare!(observer, state)
                for t in 0:5
                    fill_state!(state, t)
                    observe!(observer, state, t, Float64(t))
                end
                finalize!(observer, state)

                JLD2.jldopen(path, "r") do file
                    @test file["tracked_obs"] == tracked_obs
                    @test file["tracked_times"] == tracked_times
                    for (idx, t) in enumerate(tracked_times)
                        fill_state!(state, t)
                        @test file["h/$idx"] ≈ Array(state.h)
                        @test file["mdot/$idx"] ≈ Array(state.mdot)
                        @test file["total_time/$idx"] ≈ Float64(t)
                    end
                end
            end

            @testset "CSV" begin
                path = joinpath(dir, "out.csv")
                observer = IOObserver(tracked_obs, tracked_times, CSVFileWriter(), path)
                prepare!(observer, state)
                for t in 0:5
                    fill_state!(state, t)
                    observe!(observer, state, t, Float64(t))
                end
                finalize!(observer, state)

                rows = CSV.File(path)
                @test length(rows) == length(tracked_times)
                for (idx, t) in enumerate(tracked_times)
                    fill_state!(state, t)
                    row = rows[idx]
                    @test row.t == t
                    @test row.total_time ≈ Float64(t)
                    @test row.h_min ≈ minimum(state.h)
                    @test row.h_max ≈ maximum(state.h)
                    @test row.h_mean ≈ mean(Array(state.h))
                    @test row.mdot_min ≈ minimum(state.mdot)
                    @test row.mdot_max ≈ maximum(state.mdot)
                    @test row.mdot_mean ≈ mean(Array(state.mdot))
                end
            end

        end

        @testset "LiveObserver" begin
            observer = LiveObserver(tracked_obs, tracked_times)
            prepare!(observer, state)
            for t in 0:5
                fill_state!(state, t)
                observe!(observer, state, t, Float64(t))
            end
            for (idx, t) in enumerate(tracked_times)
                fill_state!(state, t)
                @test observer.history["h"][:, :, idx] ≈ Array(state.h)
                @test observer.history["mdot"][:, :, idx] ≈ Array(state.mdot)
            end
        end

        @testset "NoObserver" begin
            observer = NoObserver()
            @test prepare!(observer, state) === nothing
            @test observe!(observer, state, 0, 0.0) === nothing
            @test finalize!(observer, state) === nothing
        end

    end
