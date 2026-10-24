    @testset "Hydraulic potential (SheetTopPotential, GlaDS-2 Eq. 7)" begin

        nx, ny = 6, 6
        grid = Grid(nx, ny, 1e3, 1e3)
        sl = RegularizedCoulombSlidingLaw(0.25)

        base_mask() = begin
            mask = fill(GROUNDED, nx, ny)
            mask[end, :] .= OCEAN
            mask[1, :]   .= LAND
            mask[:, 1]   .= OTHER_BASIN
            mask
        end
        zb     = repeat(reshape(-0.02 .* grid.x, nx, 1), 1, ny)
        zs     = zb .+ 500.0
        G      = fill(0.06, nx, ny)
        taub_x = zeros(nx + 1, ny)
        taub_y = zeros(nx, ny + 1)
        dt     = 10800.0

        function make_sim(; hp = BedPotential(), cf = FilledCavities(), uniform_b = false, gap = ExplicitGapScheme(), kface = nothing)
            A_visc = fill(5e-25, nx, ny)
            b0     = uniform_b ? fill(0.01, nx, ny) : [0.01 + 0.004 * sin(i + 2j) for i in 1:nx, j in 1:ny]
            ub_x   = fill(1e-6, nx + 1, ny)
            ub_y   = zeros(nx, ny + 1)
            ieb    = fill(1e-8, nx, ny)
            p = ModelParameters(e_v = 0.0, b_min = 1e-6)
            state = State(grid)
            set_initial_conditions!(state, grid, p, sl, base_mask(), A_visc, zb, zs, b0, G, ub_x, ub_y, ieb, taub_x, taub_y)
            ps = PicardSolver(500, floattype(1e-12), CholeskyDirectSolver(grid), grid)
            return Simulation(grid, state, 1, floattype(dt), p, ConstantMeltInput(), sl; gap_scheme = gap, uplift = NoUplift(), ps = ps,
                              cavity_filling = cf, hydraulic_potential = hp, k_face_choice = something(kface, cf isa UnfilledCavities ? "upwind" : "arithmetic"))
        end
        grounded(s) = Array(s.mask) .== GROUNDED

        # The expected jump across a face between cells 1 and 2 (2 at the higher index): b2 - b1 between two GROUNDED cells,
        # the grounded cell's +-b on a face to an OCEAN/LAND boundary (whose side counts as 0), 0 elsewhere.
        side(m) = m == GROUNDED || m == OCEAN || m == LAND
        expected_jump(m1, m2, b1, b2) = side(m1) && side(m2) && (m1 == GROUNDED || m2 == GROUNDED) ?
                                        (m2 == GROUNDED ? b2 : 0.0) - (m1 == GROUNDED ? b1 : 0.0) : 0.0
        expected_jumps(b, m) = ([1 < i <= nx ? expected_jump(m[i-1, j], m[i, j], b[i-1, j], b[i, j]) : 0.0 for i in 1:nx+1, j in 1:ny],
                                [1 < j <= ny ? expected_jump(m[i, j-1], m[i, j], b[i, j-1], b[i, j]) : 0.0 for i in 1:nx, j in 1:ny+1])

        @testset "gap jumps between GROUNDED cells and at the outlets" begin
            for uniform_b in (false, true)
                sim = make_sim(; uniform_b = uniform_b)
                s = sim.state
                update_gap_jumps!(s, BedPotential())
                @test all(iszero, Array(s.db_x)) && all(iszero, Array(s.db_y))
                update_gap_jumps!(s, SheetTopPotential())
                b, m = Array(s.b), Array(s.mask)
                ex, ey = expected_jumps(b, m)
                @test Array(s.db_x) == ex && Array(s.db_y) == ey
                @test Array(s.db_x)[nx+1, 3] == 0                       # the domain edge
                @test Array(s.db_x)[nx, 3] == -b[nx-1, 3] < 0           # grounded -> OCEAN
                @test Array(s.db_x)[2, 3] == b[2, 3] > 0                # LAND -> grounded
                @test all(iszero, Array(s.db_y)[:, 2])                  # OTHER_BASIN -> grounded: closed, no jump
                uniform_b && @test all(iszero, Array(s.db_x)[3:nx-1, :]) # a uniform gap drives the outlets only
                uniform_b || @test any(!iszero, Array(s.db_x)[3:nx-1, :]) && any(!iszero, Array(s.db_y)[:, 3:ny])
            end
        end

        @testset "default is SheetTopPotential; BedPotential kept when passed" begin
            for cf in (FilledCavities(), UnfilledCavities())
                @test make_sim(; hp = SheetTopPotential(), cf = cf).hp isa SheetTopPotential
                @test make_sim(; hp = BedPotential(), cf = cf).hp isa BedPotential
            end
            sim = make_sim()
            default = Simulation(grid, sim.state, 1, floattype(dt), sim.p, ConstantMeltInput(), sl; gap_scheme = ExplicitGapScheme(), cavity_filling = FilledCavities(), uplift = NoUplift(), ps = sim.hs.ps)
            @test default.hp isa SheetTopPotential
        end

        @testset "flux is driven by h + b, and the solve conserves that flux" begin
            for cf in (FilledCavities(), UnfilledCavities())
                sim = make_sim(; hp = SheetTopPotential(), cf = cf)
                s = sim.state
                W_old = copy(Array(s.W))
                Shakti.prepare_head_solve!(sim); Shakti.step_h!(sim.hs, sim)
                @test first(Shakti.picard_status(sim.hs))
                h, b, m = Array(s.h), Array(s.b), Array(s.mask)
                Kx, qx, vx = Array(s.K_x), Array(s.q_x), Array(s.valid_x)
                # every face touching a grounded cell: q = -K*((h_i - h_{i-1}) + db)/dx, db = b_i - b_{i-1} between two grounded
                # cells and the grounded cell's +-b at the LAND (i = 2) and OCEAN (i = nx) outlets
                ex, _ = expected_jumps(b, m)
                for j in 2:ny, i in 2:nx
                    (m[i-1, j] == GROUNDED || m[i, j] == GROUNDED) || continue
                    @test qx[i, j] ≈ -Kx[i, j] * ((h[i, j] - h[i-1, j]) + ex[i, j]) / grid.dx * vx[i, j] rtol = 1e-12 # vx: an outflow-only face closed
                end
                @test all(vx[2, 2:ny] .== 1) && all(qx[2, 2:ny] .< 0) # the open LAND outlet drains the ice (the OCEAN faces of this fixture stay closed)
                # the converged head satisfies the cell budgets with that same flux (gap update still to come):
                # div(q) = mdot/rho_w + ieb - rate, rate = db/dt the head row assumed (explicit gap: exactly what step_b! applies)
                step_b!(sim) # the elliptic scheme: W from the water budget (filled: the gap)
                qx, qy = Array(s.q_x), Array(s.q_y)
                div = [(qx[i+1, j] - qx[i, j]) / grid.dx + (qy[i, j+1] - qy[i, j]) / grid.dy for i in 1:nx, j in 1:ny]
                dW = (Array(s.W) .- W_old) ./ dt
                res = dW .+ div .- (Array(s.mdot) ./ sim.p.rho_w .+ Array(s.ieb))
                g = grounded(s)
                scale = maximum(abs, Array(s.mdot) ./ sim.p.rho_w .+ Array(s.ieb)) + maximum(Array(s.beta) .* Array(s.abs_ub))
                @test maximum(abs, res[g]) < 1e-6 * scale
            end
        end

        @testset "the b term changes the solution and points from large to small gap" begin
            bed = make_sim(); top = make_sim(; hp = SheetTopPotential())
            foreach(step!, (bed, top))
            @test Array(top.state.h) != Array(bed.state.h)
            # at fixed head, a face across which b rises carries extra flux toward the smaller gap
            s = top.state
            s.h .= 100.0; s.b .= 0.01; s.b[3, 3] = 0.02
            update_gap_jumps!(s, SheetTopPotential()); compute_dhdxy!(s, grid); compute_face_flux!(s, top.p, top.kfs)
            @test s.q_x[4, 3] > 0  # face (3,3)-(4,3): water leaves the larger gap toward +x
            @test s.q_x[3, 3] < 0  # face (2,3)-(3,3): toward -x
        end

    end
