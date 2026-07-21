module Shakti

using Preferences
using ParallelStencil
using SparseArrays
using LinearAlgebra
using Krylov
using Base.Threads
using Statistics
using NetCDF
using HDF5
using JLD2
using CSV
using CairoMakie
using GLMakie

const backend = @load_preference("backend", "Threads")
const floattype_str = @load_preference("floattype", "Float64")

const floattype = floattype_str == "Float64" ? Float64 :
                  floattype_str == "Float32" ? Float32 :
                  error("Unknown floattype preference: $floattype_str (expected \"Float64\" or \"Float32\")")

@static if backend == "Metal"
    using Metal
    @init_parallel_stencil(Metal, floattype, 2)
elseif backend == "Threads"
    @init_parallel_stencil(Threads, floattype, 2)
else
    error("Unknown backend preference: $backend (expected \"Threads\" or \"Metal\")")
end

export backend, floattype # defined above, from the Preferences-backed backend/floattype constants

include("model_parameters.jl")
include("grid.jl")
include("state.jl")
include("mask.jl")
include("melt_input.jl")
include("k_face_scheme.jl")
include("linear_solver.jl")
include("observer.jl")
include("melt_rate.jl")
include("elliptic_solver.jl")
include("simulation.jl")
include("static_fields.jl")
include("pressure.jl")
include("field_gradients.jl")
include("water_flux.jl")
include("gap_height.jl")
include("initial_conditions.jl")
include("run.jl")
include("animation.jl")

# model_parameters.jl
export ModelParameters
export pow, canonical_exponent

# grid.jl
export Grid

# state.jl
export State

# mask.jl
export GROUNDED, OCEAN, LAND, OTHER_BASIN
export compute_face_masks!, apply_mask_to_sliding!

# melt_input.jl
export AbstractMeltInput, ConstantMeltInput, initialize_ieb!, compute_ieb!

# k_face_scheme.jl
export AbstractKFaceScheme, Arithmetic, Harmonic, compute_K_face

# linear_solver.jl
# (update_SALS!/update_MFLS! are internal assembly plumbing, not part of the
# public API; AbstractLinearSystem/SparseAssembledLinearSystem/MatrixFreeLinearSystem
# ARE public -- they're passed as the representation-choosing argument to the
# iterative solver constructors, e.g. GMRESIterativeSolver(g, MatrixFreeLinearSystem))
export AbstractLinearSolver, AbstractDirectSolver, AbstractIterativeSolver
export AbstractLinearSystem, SparseAssembledLinearSystem, MatrixFreeLinearSystem
export LUDirectSolver, GMRESIterativeSolver, BiCGSTABIterativeSolver
export solve_linear_system!

# observer.jl
export AbstractFileWriter, NetCDFFileWriter, HDF5FileWriter, JLD2FileWriter, CSVFileWriter
export AbstractObserver, NoObserver, IOObserver, LiveObserver
export get_observable
export prepare!, observe!, openfile!, write2file!, finalize!

# melt_rate.jl
export AbstractSensibleHeatScheme, WithSensibleHeat, NoSensibleHeat
export compute_taub_x!, compute_taub_y!, compute_shear!, compute_potential!, compute_sensible!, compute_mdot!

# elliptic_solver.jl
export AbstractHeadRelaxation, NoHeadRelaxation, UnderHeadRelaxation
export relax_h!
export PicardSolver
export elliptic_solver!, Picard_loop!, Picard_iteration!

# simulation.jl
export AbstractHeadScheme, ParabolicHeadScheme, EllipticHeadScheme
export AbstractGapScheme, ExplicitGapScheme, ImplicitGapScheme
export Simulation

# static_fields.jl
export compute_H!, compute_po!, compute_h!, compute_abs_ub!

# pressure.jl
export compute_pw!, compute_N!

# field_gradients.jl
export compute_dhdx!, compute_dhdy!, compute_dpwdx!, compute_dpwdy!

# water_flux.jl
export compute_q_x!, compute_q_y!, compute_Re_x!, compute_Re_y!, compute_Re!, compute_K!

# gap_height.jl
export compute_beta!, compute_b_x!, compute_b_y!, compute_b!

# initial_conditions.jl
export set_initial_conditions!

# run.jl
export run!, step!, step_h!, step_b!

# animation.jl
export make_mp4_mid, make_mp4_2d, get_moulin_ij

end