# Runs the example.jl scenario once with a chosen solver and reports the
# total run time. You control the thread count yourself (via `julia -t N`)
# and the solver via a command-line argument -- run it as many times, with
# whatever combinations, as you want. CPU-only: backend is fixed to "Threads".
#
# Usage: julia -t <threads> --project=. test/benchmark_threads_cpu.jl <solver> [nx ny]
# where <solver> is one of: LU, GMRES-matrix, GMRES-matrix-free, BiCGSTAB-matrix, BiCGSTAB-matrix-free
# nx/ny default to 128 128 if omitted.

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti

function build_example_simulation(nx, ny, tsteps, dt, ls_ctor)

    lx = 1e3
    ly = 1e3

    p = ModelParameters(e_v = 0.0)
    gap_scheme_choice = "explicit"
    k_face_choice = "arithmetic"

    grid = Grid(nx, ny, lx, ly)
    state = State(grid)
    mi = ConstantMeltInput()

    moulin_ij = (ceil(Int, nx / 2), ceil(Int, ny / 2))
    im, jm = moulin_ij
    xm, ym = grid.x[im], grid.y[jm]

    d  = 100.0
    xc = -200.0
    R  = xm - d - xc
    yc = ym

    mask = fill(GROUNDED, nx, ny)
    for j in 1:ny, i in 1:nx
        if (grid.x[i] - xc)^2 + (grid.y[j] - yc)^2 <= R^2
            mask[i, j] = OTHER_BASIN
        end
    end
    mask[end, :] .= OCEAN
    mask[:, 1]   .= OTHER_BASIN
    mask[:, end] .= OTHER_BASIN

    slope         = 0.02
    ice_thickness = 500.0
    water_depth   = 0.01
    A_visc_val    = 5e-25

    A_visc = fill(A_visc_val, nx, ny)
    zb     = repeat(reshape(-slope .* grid.x, nx, 1), 1, ny)
    zs     = zb .+ ice_thickness
    b      = fill(water_depth, nx, ny)
    G      = fill(0.06, nx, ny)
    ub_x   = fill(1e-6, nx + 1, ny)
    ub_y   = zeros(nx, ny + 1)
    ieb    = zeros(nx, ny)
    ieb[im, jm] = 3 / (grid.dx * grid.dy)

    set_initial_conditions!(state, grid, p, mi, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)

    ls = ls_ctor(grid)
    ps = PicardSolver(100, 1e-6, ls, grid; alpha = nothing)

    return Simulation(grid, state, tsteps, dt, p, gap_scheme_choice, String[], mi; ps = ps, k_face_choice = k_face_choice, verbose = true)

end

solver_ctors = Dict(
    "LU"                   => grid -> LUDirectSolver(grid),
    "GMRES-matrix"         => grid -> GMRESIterativeSolver(grid, SparseAssembledLinearSystem),
    "GMRES-matrix-free"    => grid -> GMRESIterativeSolver(grid, MatrixFreeLinearSystem),
    "BiCGSTAB-matrix"      => grid -> BiCGSTABIterativeSolver(grid, SparseAssembledLinearSystem),
    "BiCGSTAB-matrix-free" => grid -> BiCGSTABIterativeSolver(grid, MatrixFreeLinearSystem),
)

if !(length(ARGS) == 1 || length(ARGS) == 3) || !haskey(solver_ctors, ARGS[1])
    error("Usage: julia -t <threads> --project=. test/benchmark_threads_cpu.jl <solver> [nx ny]\nwhere <solver> is one of: $(join(keys(solver_ctors), ", "))\nnx/ny default to 128 128 if omitted.")
end

ls_ctor = solver_ctors[ARGS[1]]
nx, ny = length(ARGS) == 3 ? (parse(Int, ARGS[2]), parse(Int, ARGS[3])) : (128, 128)

println("Solver = $(ARGS[1]), threads = $(Threads.nthreads())")

sim = build_example_simulation(nx, ny, 20, 3600.0, ls_ctor)
elapsed = @elapsed run!(sim)

println("Run time: $(round(elapsed, digits = 3)) s")
