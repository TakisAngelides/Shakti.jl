# Makes the figures of the example pages (docs/src/figures). Not part of the docs build: the runs take
# minutes and some inputs are too big to ship. Run it as a batch job when a figure needs to change:
#
#   JULIA_LOAD_PATH="@:docs:@stdlib" julia -t 8 --project=. docs/make_figures.jl [key ...]
#
# Keys: Helheim DrangDrung Thwaites GrIS_16km GrIS_8km AIS_32km AIS_16km (no key = all). The ice-sheet
# inputs are Yelmo restarts, listed in the local, git-ignored docs/figure_paths.jl as
#   const YELMO_RUNS = [("GrIS_16km", "Greenland 16 km", "/path/to/yelmo_restart.nc"), ...]
# SHAKTI_FIG_LS = "lu" (default) or "cudss" picks the linear solver; SHAKTI_FIG_OUT is where the NetCDF
# output goes; SHAKTI_FIG_REUSE = 1 replots from the NetCDF files already there; SHAKTI_FIG_WARMUP = 1 runs one
# step first, so the wall time leaves out compilation. Each run prints one RESULT line, which is where the tables on the example pages come from.

using Shakti
using CairoMakie
using MAT, NetCDF, JLD2
using Statistics: mean, quantile, cor

const ROOT = dirname(@__DIR__)
const FIGS = mkpath(get(ENV, "SHAKTI_FIG_DIR", joinpath(@__DIR__, "src", "figures")))
const OUT  = mkpath(get(ENV, "SHAKTI_FIG_OUT", joinpath(tempdir(), "shakti_figures")))
const LS   = get(ENV, "SHAKTI_FIG_LS", "lu")
const SECONDS_PER_YEAR = 365 * 86400.0
isfile(joinpath(@__DIR__, "figure_paths.jl")) && include(joinpath(@__DIR__, "figure_paths.jl"))

# ------------------------------------------------------------------------------------------------
# Plot style: one row per field, one shared colorbar per row, LaTeX ticks, log colorbars at integer powers of ten
ticks_km   = vs -> [L"%$(round(Int, v))" for v in vs]
ticks_frac = vs -> [L"%$(round(v, digits = 2))" for v in vs]
pow10      = vs -> [L"10^{%$(round(Int, v))}" for v in vs]
sci(x) = (e = floor(Int, log10(x)); (round(x / 10.0^e, digits = 1), e))   # 2.5e-3 -> (2.5, -3)

# fields[r] = (label, column -> 2D array (NaN off the ice), colorrange or nothing, colormap, linear?);
# a log row holds log10 of the field and gets integer powers of ten spanning its 1st-99th percentile
function field_grid(path, xkm, ykm, columns, fields; size = (900, 1000), horizontal = length(columns) == 1)   # one run: fields side by side
    fig = Figure(; size, fontsize = 14)
    for (r, (name, get, crange, cmap, linear)) in enumerate(fields)
        vals = [get(c) for c in columns]
        if crange === nothing
            all = filter(isfinite, reduce(vcat, vec.(vals)))
            crange = (floor(quantile(all, 0.01)), ceil(quantile(all, 0.99)))
            crange[2] == crange[1] && (crange = (crange[1], crange[1] + 1))
        end
        local hm
        for (c, (col, v)) in enumerate(zip(columns, vals))
            fin = filter(isfinite, v)
            stat = linear ? "\\max\\ $(round(maximum(fin), digits = 2))" :
                            (s = sci(10.0^maximum(fin)); "\\max\\ $(s[1]) \\times 10^{$(s[2])}")
            title = Makie.LaTeXString("\$" * (r == 1 && !horizontal ? "\\textrm{$(col.title)},\\ " : "") * stat * "\$")
            pos = horizontal ? fig[1, r] : fig[r, c]
            ax = Axis(pos; aspect = DataAspect(), title,
                      xlabel = horizontal || r == length(fields) ? L"x\ [\mathrm{km}]" : "",
                      ylabel = (horizontal ? r : c) == 1 ? L"y\ [\mathrm{km}]" : "",
                      xtickformat = ticks_km, ytickformat = ticks_km, yticklabelsvisible = (horizontal ? r : c) == 1,
                      xticklabelsvisible = horizontal || r == length(fields))
            hm = heatmap!(ax, xkm, ykm, v; colormap = cmap, colorrange = crange, nan_color = :transparent)
        end
        Colorbar(horizontal ? fig[2, r] : fig[r, length(columns) + 1], hm; label = name, vertical = !horizontal, (horizontal ? (; width = Relative(0.85)) : (;))...,
                 tickformat = linear ? ticks_frac : pow10, ticks = linear ? (0:0.25:1) : (crange[1]:crange[2]))
    end
    colgap!(fig.layout, 8); rowgap!(fig.layout, 8)
    save(path, fig; px_per_unit = 1.5)
    println("saved ", path)
end

# The three rows every example shows, from the last frame of a run's NetCDF output
last_frame(path, name) = (v = ncread(path, name); v[:, :, end])
function standard_rows(mask)
    land(f) = ifelse.(mask .== GROUNDED, f, NaN)
    return [(L"N/p_o", c -> land(last_frame(c.path, "N") ./ last_frame(c.path, "po")), (0, 1), :viridis, true),
            (L"b\ [\mathrm{m}]", c -> land(log10.(last_frame(c.path, "b"))), nothing, :cividis, false),
            (L"\mathrm{Re}", c -> land(log10.(max.(last_frame(c.path, "Re"), 1e-3))), nothing, :magma, false)]
end

# ------------------------------------------------------------------------------------------------
# Inputs
function smooth_grounded!(f, mask, passes = 3)   # A has jumps between hardness classes; three passes of a 5-point mean
    tmp = similar(f)
    for _ in 1:passes
        tmp .= f
        for j in axes(f, 2), i in axes(f, 1)
            mask[i, j] == GROUNDED || continue
            s, n = f[i, j], 1
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                checkbounds(Bool, f, i + di, j + dj) && mask[i + di, j + dj] == GROUNDED && (s += f[i + di, j + dj]; n += 1)
            end
            tmp[i, j] = s / n
        end
        f .= tmp
    end
    return f
end

function float_to_ocean!(mask, H, zb)            # grounded cells below flotation that connect to the ocean float
    subfl = (mask .== GROUNDED) .& (917.0 .* H .< 1027.0 .* max.(0.0, -zb))
    nbrs = CartesianIndex.(((1, 0), (-1, 0), (0, 1), (0, -1)))
    queue = [I for I in findall(subfl) if any(d -> checkbounds(Bool, mask, I + d) && mask[I + d] == OCEAN, nbrs)]
    seen = falses(size(mask)); seen[queue] .= true
    while !isempty(queue)
        I = pop!(queue)
        for d in nbrs
            J = I + d
            checkbounds(Bool, mask, J) && subfl[J] && !seen[J] && (seen[J] = true; push!(queue, J))
        end
    end
    mask[seen] .= OCEAN
    return mask
end

function faces(vx, vy)                           # cell-centred speeds [m/s] onto the x- and y-faces
    Nx, Ny = size(vx)
    ub_x = zeros(Nx + 1, Ny); ub_y = zeros(Nx, Ny + 1)
    ub_x[1, :] .= vx[1, :]; ub_x[end, :] .= vx[end, :]; ub_x[2:Nx, :] .= (vx[1:end-1, :] .+ vx[2:end, :]) ./ 2
    ub_y[:, 1] .= vy[:, 1]; ub_y[:, end] .= vy[:, end]; ub_y[:, 2:Ny] .= (vy[:, 1:end-1] .+ vy[:, 2:end]) ./ 2
    return ub_x, ub_y
end

# Barycentric interpolation of vertex fields (and element fields, taken as constant) from an ISSM mesh onto a grid
function rasterize(xs, ys, el, xc, yc; vertex = (), element = ())
    Nx, Ny, DX, DY = length(xc), length(yc), xc[2] - xc[1], yc[2] - yc[1]
    out_v = [fill(NaN, Nx, Ny) for _ in vertex]; out_e = [fill(NaN, Nx, Ny) for _ in element]
    inside = falses(Nx, Ny)
    for t in axes(el, 1)
        i1, i2, i3 = el[t, :]
        xa, ya, xb, yb, xd, yd = xs[i1], ys[i1], xs[i2], ys[i2], xs[i3], ys[i3]
        den = (yb - yd) * (xa - xd) + (xd - xb) * (ya - yd)
        abs(den) < 1e-9 && continue
        ilo = max(1, floor(Int, (min(xa, xb, xd) - xc[1]) / DX) + 1); ihi = min(Nx, ceil(Int, (max(xa, xb, xd) - xc[1]) / DX) + 1)
        jlo = max(1, floor(Int, (min(ya, yb, yd) - yc[1]) / DY) + 1); jhi = min(Ny, ceil(Int, (max(ya, yb, yd) - yc[1]) / DY) + 1)
        for j in jlo:jhi, i in ilo:ihi
            w1 = ((yb - yd) * (xc[i] - xd) + (xd - xb) * (yc[j] - yd)) / den
            w2 = ((yd - ya) * (xc[i] - xd) + (xa - xd) * (yc[j] - yd)) / den
            w3 = 1 - w1 - w2
            (w1 >= -1e-9 && w2 >= -1e-9 && w3 >= -1e-9) || continue
            for (o, f) in zip(out_v, vertex); o[i, j] = w1 * f[i1] + w2 * f[i2] + w3 * f[i3]; end
            for (o, f) in zip(out_e, element); o[i, j] = f[t]; end
            inside[i, j] = true
        end
    end
    return out_v, out_e, inside
end

function read_exp_points(path)
    lines = readlines(path)
    i0 = findfirst(l -> startswith(l, "# X pos"), lines)
    pts = [parse.(Float64, split(l)) for l in lines[i0+1:end] if !isempty(strip(l))]
    return first.(pts), last.(pts)
end

function point_in_polygon(px, py, X, Y)
    inside, j = false, length(X)
    for i in eachindex(X)
        ((Y[i] > py) != (Y[j] > py)) && (px < (X[j] - X[i]) * (py - Y[i]) / (Y[j] - Y[i]) + X[i]) && (inside = !inside)
        j = i
    end
    return inside
end

# Helheim: the ISSM mesh of Sommers et al. (2023) on a 500 m grid; the terminus is open (OCEAN)
function load_helheim(; DX = 500.0)
    dir = joinpath(@__DIR__, "src", "examples", "input", "Helheim")
    md  = matread(joinpath(dir, "Helheim_winter_A0_1d.mat"))["md"].d
    ms  = md["mesh"].d
    xs, ys, el = vec(ms["x"]), vec(ms["y"]), Int.(ms["elements"])
    C_in = vec(matread(joinpath(dir, "friction_coefficient_Nfinal.mat"))["friction_coefficient"])
    x0, x1 = extrema(xs); y0, y1 = extrema(ys)
    xc = collect(x0:DX:x1); yc = collect(y0:DX:y1)
    (H, zb, A, G, VX, VY, C), _, inside = rasterize(xs, ys, el, xc, yc; vertex = (
        vec(md["geometry"].d["thickness"]), vec(md["geometry"].d["bed"]),
        vec(md["materials"].d["rheology_B"]) .^ -3.0,            # Glen: A = B^(-n)
        vec(md["basalforcings"].d["geothermalflux"]),
        vec(md["initialization"].d["vx"]), vec(md["initialization"].d["vy"]), C_in))
    mask = ifelse.(inside, GROUNDED, OTHER_BASIN)
    for f in (H, zb, A, G, VX, VY, C); f[.!inside] .= 0.0; end

    # boundary cells near the mesh's boundary vertices inside terminus13.exp drain to the ocean
    tx, ty = read_exp_points(joinpath(dir, "terminus13.exp"))
    onb = vec(ms["vertexonboundary"]) .> 0.5
    term = [v for v in eachindex(xs) if onb[v] && point_in_polygon(xs[v], ys[v], tx, ty)]
    m0 = copy(mask)
    for j in axes(mask, 2), i in axes(mask, 1)
        m0[i, j] == GROUNDED || continue
        edge = i == 1 || i == size(mask, 1) || j == 1 || j == size(mask, 2) ||
               OTHER_BASIN in (m0[i-1, j], m0[i+1, j], m0[i, j-1], m0[i, j+1])
        edge && any(v -> hypot(xc[i] - xs[v], yc[j] - ys[v]) <= 1.5 * DX, term) && (mask[i, j] = OCEAN)
    end
    ub_x, ub_y = faces(VX ./ SECONDS_PER_YEAR, VY ./ SECONDS_PER_YEAR)
    return (; xc, yc, DX, mask, H, zb, A, G, ub_x, ub_y, C)
end

# Drang Drung: the rasterized v2 data of Thota et al. (2026) (no sliding, no surface input)
function load_drangdrung(; DX = 100)
    f = jldopen(joinpath(ROOT, "test", "drangdrung", "data_v2", "DRANGDRUNG$(DX)m_v2.jld2"), "r")
    H, zb, A, G, C, mask, xc, yc = f["H"], f["B"], f["A"], f["G"], f["FRIC"], Int.(f["mask"]), f["xc"], f["yc"]
    close(f)
    Nx, Ny = size(H)
    return (; xc, yc, DX = Float64(DX), mask, H, zb, A, G, ub_x = zeros(Nx + 1, Ny), ub_y = zeros(Nx, Ny + 1), C)
end

# ISSM's own Drang Drung solution (the .mat's last frame) on the same grid: N, gap and Re
function load_drangdrung_issm(xc, yc)
    md = matread(joinpath(ROOT, "test", "drangdrung", "data_v2", "DD_SHAKTI_winter_1yr.mat"))["md"].d
    ms = md["mesh"].d
    ts = md["results"]["TransientSolution"]
    val(name) = vec(ts.values[findfirst(==(name), ts.names)][1, end])
    (N,), (b, Re), _ = rasterize(vec(ms["x"]), vec(ms["y"]), Int.(ms["elements"]), xc, yc;
                                 vertex = (val("EffectivePressure"),), element = (val("HydrologyGapHeight"), val("HydrologyReynolds")))
    return (; N, b, Re, day = round(Int, only(ts.values[findfirst(==("time"), ts.names)][1, end]) * 365))
end

# Thwaites: the 2 km grid of Kazmierczak et al. (2024), with their inverted friction (regularized Coulomb, constant v0)
function load_thwaites()
    dir = joinpath(ROOT, "test", "thwaites", "data")
    f = jldopen(joinpath(dir, "THWAITES2km_m3_HAB_toto_trimmed.jld2"), "r")
    H, zb, M, A, G, vx, vy = f["H"], f["B"], f["MASKo"], f["A"], f["G"], f["uxssa"], f["uyssa"]
    close(f)
    C, v0 = jldopen(g -> (g["C"], g["v0"]), joinpath(dir, "K24_friction_coefficient_C_HARD.jld2"), "r")
    # swap x and y so the sea (the bottom edge of the data) is on the left
    swap(f) = f isa AbstractMatrix ? permutedims(f) : f
    H, zb, M, A, G, C, v0 = swap.((H, zb, M, A, G, C, v0))
    vx, vy = swap(vy), swap(vx)
    mask = map(m -> m == 1 ? GROUNDED : m == -1 ? OTHER_BASIN : OCEAN, M)
    mask[(mask .== GROUNDED) .& (H .== 0)] .= OTHER_BASIN
    float_to_ocean!(mask, H, zb)
    DX = 2000.0
    xc, yc = DX .* (0:size(H, 1)-1), DX .* (0:size(H, 2)-1)
    ub_x, ub_y = faces(vx ./ SECONDS_PER_YEAR, vy ./ SECONDS_PER_YEAR)
    A = smooth_grounded!(A ./ SECONDS_PER_YEAR, mask)                  # stored per year
    return (; xc, yc, DX, mask, H, zb, A, G, ub_x, ub_y, C, v0 = v0 / SECONDS_PER_YEAR)
end

# A Yelmo restart (Greenland or Antarctica)
function load_yelmo(path)
    xc, yc = Float64.(ncread(path, "xc")) .* 1e3, Float64.(ncread(path, "yc")) .* 1e3   # km -> m
    rd3(v) = reshape(Float64.(ncread(path, v)), length(xc), length(yc), :)    # some restarts carry a time dimension
    rd(v) = rd3(v)[:, :, 1]
    f_ice, f_grnd, H, zb = rd("f_ice"), rd("f_grnd"), rd("H_ice"), rd("z_bed")
    mask = map((fi, fg, h, z) -> fi > 0 && fg > 0 && h > 0 ? GROUNDED : fi == 0 && z > 0 ? LAND : OCEAN, f_ice, f_grnd, H, zb)
    float_to_ocean!(mask, H, zb)
    A = smooth_grounded!(dropdims(mean(rd3("ATT"), dims = 3), dims = 3) ./ SECONDS_PER_YEAR, mask)   # 1/yr/Pa^3 -> SI
    G = rd("Q_geo") ./ 1e3                                             # mW/m2 -> W/m2
    bad = G .< 0                                                        # -9999 where the source map has no data
    G[bad] .= mean(G[.!bad .& (mask .== GROUNDED)])
    ub_x, ub_y = faces(rd("ux_b") ./ SECONDS_PER_YEAR, rd("uy_b") ./ SECONDS_PER_YEAR)
    return (; xc, yc, DX = xc[2] - xc[1], mask, H, zb, A, G, ub_x, ub_y)
end

# ------------------------------------------------------------------------------------------------
# One run: a fresh state from d, Picard with a direct solver, the last frame written to NetCDF
function run_case(key, tag, d, p, sl, dt, nsteps; b0 = 1e-3, kw = (;))
    Nx, Ny = size(d.H)
    grid = Grid(Nx, Ny, (Nx - 1) * d.DX, (Ny - 1) * d.DX)
    b = fill(b0, Nx, Ny)
    state = State(grid)
    set_initial_conditions!(state, grid, p, sl(grid), d.mask, d.A, d.zb, d.zb .+ d.H .+ b, b, d.G, d.ub_x, d.ub_y,
                            zeros(Nx, Ny), zeros(Nx + 1, Ny), zeros(Nx, Ny + 1))
    ls = LS == "cudss" ? CUDSSDirectSolver(grid; structure = :general) : LUDirectSolver(grid)
    path = joinpath(OUT, "$(key)_$(tag).nc")
    get(ENV, "SHAKTI_FIG_REUSE", "0") == "1" && isfile(path) && return path   # replot from an earlier run
    sim = Simulation(grid, state, nsteps, floattype(dt), p, ConstantMeltInput(), sl(grid);
                     ps = PicardSolver(500, 1e-6, ls, grid), observer = IOObserver(["N", "b", "Re", "po"], [0, nsteps], NetCDFFileWriter(), path), kw...)
    if get(ENV, "SHAKTI_FIG_WARMUP", "0") == "1"            # time without compilation: one step on a copy first
        st1 = deepcopy(state)
        run!(Simulation(grid, st1, 1, floattype(dt), p, ConstantMeltInput(), sl(grid);
                        ps = PicardSolver(500, 1e-6, LS == "cudss" ? CUDSSDirectSolver(grid; structure = :general) : LUDirectSolver(grid), grid), kw...))
    end
    wall = @elapsed run!(sim)
    gr = Array(state.mask) .== GROUNDED
    N, po, bb, Re = (Array(getfield(state, f))[gr] for f in (:N, :po, :b, :Re))
    println("RESULT $key $tag: $Nx x $Ny, $(count(gr)) grounded, $nsteps x $(dt / 3600) h, wall $(round(wall, digits = 1)) s, ",
            "last step converged $(sim.hs.ps.converged) in $(sim.hs.ps.last_iter) its, N/po in $(round.(extrema(N ./ po), digits = 3)), ",
            "mean N $(round(mean(N) / 1e6, digits = 3)) MPa, max b $(round(maximum(bb), sigdigits = 3)) m, max Re $(round(maximum(Re), sigdigits = 3)), ",
            "b > H in $(count(bb .> d.H[gr])) cells (H there up to $(round(maximum(d.H[gr][bb .> d.H[gr]]; init = 0.0), sigdigits = 3)) m)")
    flush(stdout)
    return path
end

kmaxes(d) = ((d.xc .- d.xc[1]) ./ 1e3, (d.yc .- d.yc[1]) ./ 1e3)

# The original SHAKTI equations: full cavities, no uplift, bed potential, explicit gap coupling
const SHAKTI_ORIGINAL = (cavity_filling = FilledCavities(), uplift = NoUplift(), hydraulic_potential = BedPotential(),
                         gap_coupling = ExplicitGapCoupling())

function helheim()
    d = load_helheim()
    # Table 2 of Sommers et al. (2023); ct = 0 because their working equation drops the sensible-heat term
    common = (rho_i = 917.0, omega = 1e-3, br = 0.0, lr = 2.0, ct = 0.0, mdot_includes_sensible = false)
    sl = grid -> LinearSlidingLaw(grid, d.C)                             # tau_b = C^2 N u_b with their inverted C
    dt, nsteps = 1800.0, 4320                                            # 90 days, their winter base state
    cols = [(title = "defaults", path = run_case("Helheim", "defaults", d, ModelParameters(; common...), sl, dt, nsteps)),
            (title = "SHAKTI", path = run_case("Helheim", "shakti", d, ModelParameters(; common..., b_min = 1e-3), sl, dt, nsteps; kw = SHAKTI_ORIGINAL))]
    field_grid(joinpath(FIGS, "Helheim.png"), kmaxes(d)..., cols, standard_rows(d.mask))
end

function drangdrung()
    d = load_drangdrung()
    # ISSM's constants (omega and ct are hard-coded in its SHAKTI), b_max = 1 m as ISSM
    common = (rho_i = 917.0, omega = 1e-3, br = 0.0, lr = 1.0, ct = 0.0)
    sl = grid -> LinearSlidingLaw(grid, d.C)
    dt, nsteps = 3600.0, 8760                                            # one year, as the ISSM run
    paths = [run_case("DrangDrung", "defaults", d, ModelParameters(; common...), sl, dt, nsteps),
             run_case("DrangDrung", "issm", d, ModelParameters(; common..., b_min = 1e-3, b_max = 1.0), sl, dt, nsteps; kw = SHAKTI_ORIGINAL)]
    issm = load_drangdrung_issm(d.xc, d.yc)
    gr = (d.mask .== GROUNDED) .& isfinite.(issm.N)
    for (tag, path) in zip(("defaults", "issm"), paths)
        println("RESULT DrangDrung $tag: corr(N, ISSM day $(issm.day)) = ", round(cor(last_frame(path, "N")[gr], issm.N[gr]), digits = 3))
    end
    po = 917.0 * 9.81 .* d.H
    land(f) = ifelse.(d.mask .== GROUNDED, f, NaN)
    get_issm = Dict("N" => issm.N, "b" => issm.b, "Re" => issm.Re, "po" => po)
    frame(c, v) = c.path === nothing ? get_issm[v] : last_frame(c.path, v)
    cols = [(title = "ISSM", path = nothing), (title = "SHAKTI", path = paths[2]), (title = "defaults", path = paths[1])]
    field_grid(joinpath(FIGS, "DrangDrung.png"), kmaxes(d)..., cols, [
        (L"N/p_o", c -> land(frame(c, "N") ./ frame(c, "po")), (0, 1), :viridis, true),
        (L"b\ [\mathrm{m}]", c -> land(log10.(frame(c, "b"))), nothing, :cividis, false),
        (L"\mathrm{Re}", c -> land(log10.(max.(frame(c, "Re"), 1e-3))), nothing, :magma, false)]; size = (720, 1000))
end

# Thwaites and the ice sheets: the defaults for one year of 1-day steps
# SHAKTI_FIG_VARIANT=shakti runs the original SHAKTI equations instead (for the timings; set SHAKTI_FIG_DIR away from docs)
function ice_sheet(key, d, sl; size = (1200, 560))
    if get(ENV, "SHAKTI_FIG_VARIANT", "defaults") == "shakti"
        path = run_case(key, "shakti", d, ModelParameters(b_min = 1e-3), sl, 86400.0, 365; kw = SHAKTI_ORIGINAL)
        field_grid(joinpath(FIGS, "$(key)_shakti.png"), kmaxes(d)..., [(title = "day 365", path = path)], standard_rows(d.mask); size)
    else
        path = run_case(key, "defaults", d, ModelParameters(), sl, 86400.0, 365)
        field_grid(joinpath(FIGS, "$key.png"), kmaxes(d)..., [(title = "day 365", path = path)], standard_rows(d.mask); size)
    end
end

thwaites() = (d = load_thwaites(); ice_sheet("Thwaites", d, grid -> RegularizedCoulombV0SlidingLaw(grid, d.C, d.v0); size = (1100, 600)))
yelmo(key) = () -> begin
    path = only(r[3] for r in YELMO_RUNS if r[1] == key)
    ice_sheet(key, load_yelmo(path), grid -> RegularizedCoulombSlidingLaw(0.25); size = startswith(key, "GrIS") ? (1000, 680) : (1200, 500))
end

# ------------------------------------------------------------------------------------------------
const MAKERS = Dict("Helheim" => helheim, "DrangDrung" => drangdrung, "Thwaites" => thwaites,
                    (k => yelmo(k) for k in ("GrIS_16km", "GrIS_8km", "AIS_32km", "AIS_16km"))...)
for key in (isempty(ARGS) ? sort(collect(keys(MAKERS))) : ARGS)
    MAKERS[key]()
end
