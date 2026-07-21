# Runs the example.jl scenario once on the Metal (GPU) backend and reports
# the total run time. Metal requires Float32, and only the matrix-free
# iterative solvers work on it -- LUDirectSolver and the matrix-based
# GMRES/BiCGSTAB variants are CPU-only (SparseArrays/UMFPACK and Krylov's
# sparse matvec have no GPU path) and error immediately if constructed here.
#
# First run pays a substantial one-time Metal shader-compilation cost (tens
# of seconds even for a tiny grid) -- that's what the warmup below is for;
# don't read it as steady-state performance.
#
# Usage: julia --project=. test/test_metal.jl <solver> [nx ny]
# where <solver> is one of: GMRES-matrix-free, BiCGSTAB-matrix-free
# nx/ny default to 128 128 if omitted.

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using Preferences
set_preferences!("Shakti", "backend" => "Metal", "floattype" => "Float32"; force = true)

using Shakti
using CairoMakie

function build_example_simulation(nx, ny, tsteps, dt, ls_ctor)

    dt = floattype(dt) # Simulation's dt must exactly match ModelParameters' float type (Float32 here), unlike Grid/ModelParameters which convert their own numeric-literal arguments internally

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
    
    mask = fill(GROUNDED, nx, ny)
    mask[end, :] .= OCEAN
    mask[1, :]   .= OTHER_BASIN
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

    # set_initial_conditions! converts these plain host arrays to the active
    # backend's array type (Data.Array) internally, so they can stay plain
    # Float64/Int CPU arrays here regardless of backend.
    set_initial_conditions!(state, grid, p, mi, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb)

    ls = ls_ctor(grid)
    ps = PicardSolver(100, 1e-4, ls, grid; alpha = nothing)

    return Simulation(grid, state, tsteps, dt, p, gap_scheme_choice, String[], mi; ps = ps, k_face_choice = k_face_choice, verbose = true)

end

solver_ctors = Dict(
    "GMRES-matrix-free"    => grid -> GMRESIterativeSolver(grid, MatrixFreeLinearSystem),
    "BiCGSTAB-matrix-free" => grid -> BiCGSTABIterativeSolver(grid, MatrixFreeLinearSystem),
)

if !(length(ARGS) == 1 || length(ARGS) == 3) || !haskey(solver_ctors, ARGS[1])
    error("Usage: julia --project=. test/test_metal.jl <solver> [nx ny]\nwhere <solver> is one of: $(join(keys(solver_ctors), ", "))\nnx/ny default to 128 128 if omitted.")
end

ls_ctor = solver_ctors[ARGS[1]]
nx, ny = length(ARGS) == 3 ? (parse(Int, ARGS[2]), parse(Int, ARGS[3])) : (128, 128)

println("Solver = $(ARGS[1]), backend = $(backend), floattype = $(floattype)")

dt = 3600.0
tsteps = 39
sim = build_example_simulation(nx, ny, tsteps, dt, ls_ctor)
elapsed = @elapsed run!(sim)

println("Run time: $(round(elapsed, digits = 3)) s")

# Plot the final state regardless of whether Picard converged to the set
# tol -- useful for eyeballing whether the physics still looks sane even on
# a run that didn't fully converge. Array(...) pulls GPU-resident fields
# back to the host for plotting.
#
# A run that truly diverged (not just "not converged to tol") can leave
# NaN/Inf in the field, and Makie's automatic colorbar tick formatter
# crashes outright on that rather than just rendering it oddly -- so the
# colorrange is computed from finite values only, and a field with no
# finite values at all is reported instead of attempting to plot it.
function plot_field!(ax, x, y, field; label)
    finite_vals = filter(isfinite, vec(field))
    if isempty(finite_vals)
        println("  $label: entirely non-finite (NaN/Inf) -- skipping heatmap")
        return nothing
    end
    lo, hi = extrema(finite_vals)
    lo == hi && (hi = lo + max(abs(lo), one(lo)) * 1f-3) # avoid a degenerate zero-width colorrange -- Makie stores colorrange as Float32 internally, so eps(lo) alone rounds straight back to lo == hi
    n_bad = count(!isfinite, field)
    n_bad > 0 && println("  $label: $n_bad / $(length(field)) cells are non-finite (shown blank)")
    return CairoMakie.heatmap!(ax, x, y, field; colorrange = (lo, hi))
end

CairoMakie.activate!()
fig = Figure(size = (1200, 600))
ax1 = Axis(fig[1, 1], title = "Gap Height (m)", xlabel = "x (m)", ylabel = "y (m)")
hm_b = plot_field!(ax1, sim.grid.x, sim.grid.y, Array(sim.state.b); label = "b")
hm_b !== nothing && Colorbar(fig[1, 2], hm_b)
ax2 = Axis(fig[1, 3], title = "Head (m)", xlabel = "x (m)", ylabel = "y (m)")
hm_h = plot_field!(ax2, sim.grid.x, sim.grid.y, Array(sim.state.h); label = "h")
hm_h !== nothing && Colorbar(fig[1, 4], hm_h)

dir = joinpath(@__DIR__, "Figures")
mkpath(dir)
outfile = joinpath(dir, "final_state_metal.png")
save(outfile, fig)
display(fig)

println("Saved final-state plot to $outfile")
