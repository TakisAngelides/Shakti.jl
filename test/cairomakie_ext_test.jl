# The CairoMakie weak dependency: `using CairoMakie` must activate the extension that provides the movie functions.
using CairoMakie
@testset "CairoMakie extension" begin
    ext = Base.get_extension(Shakti, :ShaktiCairoMakieExt)
    @test ext !== nothing
    @test !isempty(methods(Shakti.make_mp4_mid))
    @test !isempty(methods(Shakti.make_mp4_2d))
    @test all(m -> m.module === ext, filter(m -> m.sig <: Tuple{typeof(Shakti.make_mp4_mid),AbstractArray,Vararg}, collect(methods(Shakti.make_mp4_mid))))
end
