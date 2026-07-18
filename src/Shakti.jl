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

export backend, floattype

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

export Grid
export ModelParameters
export pow, canonical_exponent
export State

export GROUNDED, OCEAN, LAND, OTHER_BASIN
export compute_face_masks!, apply_mask_to_sliding!

export AbstractLinearSolver, AbstractDirectSolver, AbstractIterativeSolver
export CholeskySolver, LUSolver, KrylovCGSolver, KrylovGMRESSolver
export solve!, update!, solve_linear_system!

export AbstractHeadRelaxation, NoHeadRelaxation, UnderHeadRelaxation
export PicardSolver

export AbstractMeltInput, ConstantMeltInput, initialize_ieb!, compute_ieb!

export AbstractKFaceScheme, Arithmetic, Harmonic, compute_K_face

export AbstractFileWriter, NetCDFFileWriter, HDF5FileWriter, JLD2FileWriter, CSVFileWriter
export AbstractObserver, NoObserver, IOObserver, LiveObserver
export get_observable

export AbstractSensibleHeatScheme, WithSensibleHeat, NoSensibleHeat

export AbstractHeadScheme, ParabolicHeadScheme, EllipticHeadScheme
export AbstractGapScheme, ExplicitGapScheme, ImplicitGapScheme
export Simulation

export elliptic_solver!, Picard_loop!, Picard_iteration!, relax_h!
export prepare!, observe!, openfile!, write2file!, finalize!

export compute_H!, compute_po!, compute_h!, compute_abs_ub!
export compute_dhdx!, compute_dhdy!, compute_dpwdx!, compute_dpwdy!
export compute_pw!, compute_N!
export compute_q_x!, compute_q_y!
export compute_Re_x!, compute_Re_y!, compute_Re!
export compute_taub_x!, compute_taub_y!
export compute_shear!, compute_potential!, compute_sensible!, compute_mdot!
export compute_K!
export compute_beta!, compute_b_x!, compute_b_y!, compute_b!

export run!, step!, step_h!, step_b!
export set_initial_conditions!

export make_mp4_mid, make_mp4_2d, get_moulin_ij

end