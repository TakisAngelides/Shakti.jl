# [Performance](@id Performance)

Wall times of the example runs on the AWI cluster albedo, from `docs/make_figures.jl`.

* GPU: one NVIDIA A100, `CUDSSDirectSolver(grid; structure = :general)`.
* CPU: 8 threads on a 128-core node, `LUDirectSolver(grid)`.
* Both with `PicardSolver(500, 1e-6, ...)` and four output fields written at the first and last step.
* Total is the wall time of `run!` for the whole simulation, all steps; per step is total divided by
  the number of steps. Compilation is excluded: each run does a one-step warm-up first.

Options:

* defaults: `UnfilledCavities()`, `IceUplift()`, `SheetTopPotential()`, `"implicit_upwind"` faces,
  `ConsistentGapCoupling()`, `b_min = 1e-6`, `b_max = Inf`.
* SHAKTI: the original equations, `FilledCavities()`, `NoUplift()`, `BedPotential()`,
  `ExplicitGapCoupling()`, `b_min = 1e-3` (Drang Drung also `b_max = 1`), and `"arithmetic"` faces
  (the default without unfilled cavities or uplift).
* Helheim and Drang Drung use the physical constants of their ISSM runs (see their pages); the ice
  sheets use the default constants.

| Example | Options | Grid | Grounded cells | Steps | `dt` | Simulated | GPU total | GPU per step | CPU total | CPU per step |
|---|---|---|---|---|---|---|---|---|---|---|
| Greenland 16 km | defaults | 106 x 181 | 6 612 | 365 | 1 day | 1 year | 8.2 s | 22 ms | 50.5 s | 138 ms |
| Greenland 16 km | SHAKTI ¹ | 106 x 181 | 6 612 | 365 | 1 day | 1 year | 6.2 s | 17 ms | 56.8 s | 156 ms |
| Greenland 8 km | defaults | 211 x 361 | 27 601 | 365 | 1 day | 1 year | 24.8 s | 68 ms | 4.4 min | 720 ms |
| Greenland 8 km | SHAKTI ¹ | 211 x 361 | 27 601 | 365 | 1 day | 1 year | 24.5 s | 67 ms | 9.2 min | 1.5 s |
| Antarctica 32 km | defaults | 191 x 191 | 11 739 | 365 | 1 day | 1 year | 12.6 s | 35 ms | 1.7 min | 287 ms |
| Antarctica 32 km | SHAKTI ¹ | 191 x 191 | 11 739 | 365 | 1 day | 1 year | 11.7 s | 32 ms | 3.5 min | 577 ms |
| Antarctica 16 km | defaults | 381 x 381 | 47 515 | 365 | 1 day | 1 year | 57.5 s | 158 ms | 12.0 min | 2.0 s |
| Antarctica 16 km | SHAKTI ¹ | 381 x 381 | 47 515 | 365 | 1 day | 1 year | 27.5 s | 75 ms | 6.7 min | 1.1 s |
| Thwaites | defaults | 288 x 345 | 51 672 | 365 | 1 day | 1 year | 49.1 s | 135 ms | 10.5 min | 1.7 s |
| Thwaites | SHAKTI ¹ | 288 x 345 | 51 672 | 365 | 1 day | 1 year | 16.6 s | 45 ms | 3.4 min | 567 ms |
| Drang Drung | defaults | 102 x 200 | 2 695 | 8 760 | 1 h | 1 year | 1.1 min | 7.8 ms | 8.0 min | 55 ms |
| Drang Drung | SHAKTI | 102 x 200 | 2 695 | 8 760 | 1 h | 1 year | 41.4 s | 4.7 ms | 5.2 min | 36 ms |
| Helheim | defaults | 134 x 104 | 8 199 | 4 320 | 30 min | 90 days | 4.3 min | 60 ms | 42.6 min | 592 ms |
| Helheim | SHAKTI | 134 x 104 | 8 199 | 4 320 | 30 min | 90 days | 18.0 s | 4.2 ms | 3.2 min | 44 ms |

¹ Timings only. On the ice sheets the original equations do not give a usable state after a year:
where the water cannot fill the gap `p_w` falls far below zero (`N/p_o` up to 400 on Thwaites and
37 000 on Greenland 16 km), and on Greenland and Antarctica the gap runs away to 6 to 109 m. The
defaults keep `0 ≤ p_w ≤ p_o` and gaps below 1.1 m.

* The cost is set by the number of Picard iterations a step, times one sparse direct solve of the
  head system. On the ice sheets the defaults need two to seven iterations a step at the end of
  the year.
* Helheim with the defaults is the expensive case: more than ten iterations a step against one for
  the original equations.
* The GPU is 6 to 23 times faster than the CPU here, more on the larger grids.
