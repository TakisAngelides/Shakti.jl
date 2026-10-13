using Preferences
set_preferences!("Shakti", "backend" => "Threads", "floattype" => "Float64"; force = true)

using Shakti
using Test
using LinearAlgebra
using SparseArrays
using Statistics
using NetCDF
using HDF5
using JLD2
using CSV

@testset "Shakti.jl" begin
    include("linear_solver_test.jl")
    include("observer_test.jl")
    include("checkpoint_test.jl")
    include("sliding_law_test.jl")
    include("parabolic_solver_test.jl")
    include("frozen_bed_test.jl")
    include("gap_height_test.jl")
    include("unfilled_cavities_test.jl")
    include("hydraulic_potential_test.jl")
    include("ice_uplift_test.jl")
    include("implicit_upwind_test.jl")
    include("diffusion_test.jl")
    include("consistency_test.jl")
    include("ieb_external_test.jl")
    include("cairomakie_ext_test.jl")
end
