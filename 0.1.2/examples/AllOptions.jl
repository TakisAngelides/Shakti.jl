#=
# [All options](@id AllOptions)
A reference listing of every user-facing choice in Shakti, gathered in one place and grouped by
the source file that defines it -- for "what are ALL the choices for X", skim the section headed
with X's source file. Unlike the other examples, this one isn't chasing a particular physical
result: it builds one small, fast-running simulation, but comments every alternative not taken
right where the choice is made.
=#

using Shakti
using CairoMakie

# ## 1. Backend / float type (`src/Shakti.jl`)
# These are Julia *preferences*, baked into Shakti at `using Shakti` time via `LocalPreferences.toml`
# -- not a runtime argument, so they can't be switched inside a running session (this page's `using
# Shakti` already fixed them for everything below). Set them *before* `using Shakti`, e.g.:
# ```julia
# using Preferences
# set_preferences!("Shakti", "backend" => "CUDA", "floattype" => "Float32"; force = true)
# using Shakti
# ```
#   backend:   "Threads" (CPU, multithreaded -- start Julia with `-t N`) | "Metal" (Apple GPU) | "CUDA" (NVIDIA GPU)
#   floattype: Float64 | Float32 (GPU backends commonly use Float32 for throughput; CholeskyDirectSolver/
#              CGIterativeSolver(..., SparseAssembledLinearSystem) are CPU-only and need Float64)
println("This page was built with backend = \"$(Shakti.backend)\", floattype = $(Shakti.floattype)")

# ## 2. Grid (`src/grid.jl`)
# `Grid(nx, ny, lx, ly)`: `nx*ny` cells over a domain of size `lx*ly` (meters); `dx`/`dy` and the
# cell-center coordinate vectors `x`/`y` are derived from these four numbers.
const NX, NY = 24, 24
const LX, LY = 2000.0, 2000.0
grid = Grid(NX, NY, LX, LY)

# ## 3. Model parameters (`src/model_parameters.jl`)
# Every [`ModelParameters`](@ref) keyword, with its default. `e_v == 0.0` selects
# [`EllipticHeadScheme`](@ref) (Picard iteration, used below); `e_v != 0.0` selects
# [`ParabolicHeadScheme`](@ref) instead (single backward-Euler solve per step -- see the
# [Parabolic head scheme](@ref ParabolicScheme) example). `ct == 0.0` or `cw == 0.0` disables the
# sensible-heat term ([`NoSensibleHeat`](@ref) instead of [`WithSensibleHeat`](@ref), decided once
# in `Simulation`'s constructor). `br == 0.0` disables the opening-by-sliding term
# ([`NoOpenBySliding`](@ref) instead of [`WithOpenBySliding`](@ref), same "decided once" idiom).
p = ModelParameters(
    rho_w  = 1000.0, # density of subglacial/fresh water
    rho_sw = 1027.0, # density of ocean (sea) water -- only used for the OCEAN Dirichlet BC's hydrostatic pressure
    rho_i  = 910.0,  # density of ice
    g      = 9.81,   # gravitational acceleration
    nu     = 1.787e-6, # kinematic viscosity of water
    n      = 3.0,    # Glen's flow law exponent
    omega  = 1e-4,   # laminar/turbulent transition parameter (Table 2, Sommers et al. 2018, gives 1e-3; see ModelParameters' own docstring for why 1e-4 is the default here instead)
    L      = 334e3,  # latent heat of fusion
    br     = 0.05,   # bedrock bump height (0.0 -> NoOpenBySliding)
    lr     = 2.0,    # bedrock bump spacing
    ct     = 7.5e-8, # change of pressure-melting-point with temperature (0.0 -> NoSensibleHeat)
    cw     = 4.22e3, # heat capacity of water (0.0 -> NoSensibleHeat)
    p_atm  = 0.0,    # atmospheric pressure, the Dirichlet reference for LAND/OCEAN boundary conditions
    b_min  = 1e-3,   # minimum water thickness (numerical floor)
    e_v    = 0.0,    # englacial storage void ratio (0.0 -> EllipticHeadScheme; nonzero -> ParabolicHeadScheme)
)

# ## 4. Mask (`src/mask.jl`)
# Every cell is one of four [`State`](@ref).mask values -- [`GROUNDED`](@ref) (dynamic hydrology
# solved here), [`OCEAN`](@ref)/[`LAND`](@ref) (Dirichlet head boundaries, differing only in which
# density sets the prescribed pressure), or [`OTHER_BASIN`](@ref) (frozen; any face touching it is
# zeroed in every gradient/flux). See the [Mask variants](@ref MaskVariants) example for full
# geometries built from these four values.
mask = fill(GROUNDED, NX, NY)
mask[end, :] .= OCEAN       # Dirichlet, pw = p_atm - rho_sw*g*min(zb, 0)
mask[1, :]   .= OTHER_BASIN # frozen
mask[:, 1]   .= OTHER_BASIN
mask[:, end] .= OTHER_BASIN

# ## 5. Melt input (`src/melt_input.jl`)
# Multiple dispatch on the [`AbstractMeltInput`](@ref) subtype passed to `Simulation`/`step!`:
#   - [`ConstantMeltInput`](@ref)`()` -- `state.ieb` fixed at whatever `set_initial_conditions!` seeded, for the whole run
#   - [`SeasonalMeltInput`](@ref)`(; t_start, t_end, amplitude, offset, i_min, seconds_per_year)` -- cosine-shaped melt season, see the [Seasonal melt input](@ref SeasonalMeltInput) example
mi = ConstantMeltInput()

# ## 6. Sliding law (`src/sliding_law.jl`)
# Multiple dispatch on the [`AbstractSlidingLaw`](@ref) subtype, decided once per `Simulation`:
#   - [`RegularizedCoulombSlidingLaw`](@ref)`(C)` -- taub -> C*N as ub/N^n*lambda -> infinity, recomputed from N/ub every Picard iteration
#   - [`LinearSlidingLaw`](@ref)`(grid, C)` -- taub = C^2*N*u_b exactly; `C` can be a scalar or an `(nx,ny)` field (e.g. an inverted per-cell drag coefficient, see the [Helheim Glacier](@ref Helheim) example)
#   - [`PrescribedSlidingLaw`](@ref)`()` -- taub_x/taub_y fixed once (via `set_initial_conditions!`'s own `taub_x`/`taub_y` arguments) and never recomputed
sl = RegularizedCoulombSlidingLaw(0.25)

# ## 7. K-face scheme (`src/k_face_scheme.jl`)
# How transmissivity `K` is averaged onto a cell face from its two neighbouring cell centers,
# passed as `k_face_choice` to `Simulation`:
#   - `"arithmetic"` -> [`Arithmetic`](@ref): `(K1 + K2) / 2`
#   - `"harmonic"`   -> [`Harmonic`](@ref): `2*K1*K2 / (K1 + K2 + eps)` (weights the lower-K side more heavily)
k_face_choice = "arithmetic"

# ## 8. Gap scheme (`src/simulation.jl` / `src/gap_height.jl`)
# How gap height `b` is time-integrated, passed as `gap_scheme_choice` to `Simulation`:
#   - `"implicit"` -> [`ImplicitGapScheme`](@ref): backward-Euler on the creep-closure term, unconditionally stable, the usual choice
#   - `"explicit"` -> [`ExplicitGapScheme`](@ref): forward-Euler, cheaper per step but only stable for small enough `dt`
gap_scheme_choice = "implicit"

# ## 9. Linear solver (`src/linear_solver.jl`, `src/preconditioner.jl`)
# Passed as `ps`/`ls` to `Simulation` depending on `p.e_v` (see section 3 above):
#   - [`CholeskyDirectSolver`](@ref)`(grid)` -- CPU-only ("Threads" backend only), direct factorization
#   - [`CGIterativeSolver`](@ref)`(grid, SparseAssembledLinearSystem; chebyshev_degree, chebyshev_nsteps_estimate, amg)` -- CPU-only
#   - [`CGIterativeSolver`](@ref)`(grid, MatrixFreeLinearSystem; chebyshev_degree, chebyshev_nsteps_estimate)` -- works on every backend, the only choice compatible with Metal/CUDA
# `chebyshev_degree`: `nothing` -> plain Jacobi preconditioning; `Int >= 1` -> [`ChebyshevPreconditioner`](@ref) of that degree.
# `amg` (`SparseAssembledLinearSystem` only): `true` -> [`AMGPreconditioner`](@ref) (CPU/SparseMatrixCSC-only, the default there, and the fastest choice measured across every grid size tested); `false` -> plain Jacobi.
ls = CholeskyDirectSolver(grid)

# ## 10. Picard solver (`src/elliptic_solver.jl`)
# Only used under [`EllipticHeadScheme`](@ref) (`p.e_v == 0`). [`PicardSolver`](@ref)`(iters, tol,
# ls, grid; alpha, check_every)`:
#   - `iters`/`tol` -- maximum iterations and relative convergence tolerance
#   - `alpha`: `nothing` -> [`NoHeadRelaxation`](@ref); `Float64` in `(0, 1]` -> [`UnderHeadRelaxation`](@ref)`(alpha)`, damping oscillations on stiff problems at the cost of slower convergence
#   - `check_every`: how many Picard iterations between convergence checks (each check forces a GPU->CPU sync on GPU backends; `1` was fastest at every grid size measured so far)
ps = PicardSolver(500, 1e-6, ls, grid; alpha = nothing, check_every = 1)

# ## 11. Observer (`src/observer.jl`)
# Passed as `which_observer`/`which_file_writer` to `Simulation`:
#   - `"Live"` -> [`LiveObserver`](@ref): history kept in RAM (needed for `make_mp4_2d`/`make_mp4_mid`)
#   - `"IO"`   -> [`IOObserver`](@ref): streamed to disk via `which_file_writer`, one of `"NetCDF"`/`"HDF5"`/`"JLD2"` (full tracked-field grids per tracked time) or `"CSV"` (one row per tracked time, min/max/mean per field)
#   - `tracked_obs = String[]` (empty) -> [`NoObserver`](@ref), regardless of `which_observer`
which_observer = "Live"
tracked_obs = ["h", "b", "N", "Re", "mdot"] # any State field name (src/state.jl)
tracked_times = 0:10

# ## 12. Building and running the `Simulation` (`src/simulation.jl`, `src/run.jl`)
# Every choice above comes together here. `run!` also accepts `checkpoint_every`/`checkpoint_path`
# (periodic [`save_checkpoint`](@ref)) and `restart_path` (resume via [`load_checkpoint!`](@ref)) --
# see `src/checkpoint.jl`.
A_visc = fill(5e-25, NX, NY)
zb     = zeros(NX, NY)
zs     = zb .+ 500.0
b0     = fill(0.01, NX, NY)
G      = fill(0.06, NX, NY)
ub_x   = fill(1e-6, NX + 1, NY)
ub_y   = zeros(NX, NY + 1)
ieb    = fill(1.0 / (365 * 86400.0), NX, NY) # 1 m/yr distributed input -> m/s
taub_x = zeros(NX + 1, NY) # unused: RegularizedCoulombSlidingLaw recomputes taub from N/ub every Picard iteration
taub_y = zeros(NX, NY + 1)

state = State(grid)
set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)

sim = Simulation(grid, state, 10, floattype(3600.0), p, gap_scheme_choice, tracked_obs, mi, sl;
                 ps = ps, which_observer = which_observer, tracked_times = tracked_times,
                 k_face_choice = k_face_choice, verbose = false)
run!(sim)

fig = Figure(size = (500, 400))
ax = Axis(fig[1, 1], title = "Gap height (m) after 10 steps", aspect = DataAspect())
hm = heatmap!(ax, grid.x, grid.y, Array(sim.state.b))
Colorbar(fig[1, 2], hm)
fig
