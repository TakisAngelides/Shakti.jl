"""
$(TYPEDSIGNATURES)

A regular Cartesian grid: `nx * ny` cells spanning a domain of size `lx * ly`, with cell-center
coordinate vectors `x`/`y` starting at the origin. `dx2`/`dy2` are `dx^2`/`dy^2`, precomputed
once since they appear in every finite-difference/finite-volume stencil in the solver.
"""
struct Grid{F <: AbstractFloat, A <: AbstractArray}
    nx::Int
    ny::Int
    lx::F
    ly::F
    dx::F
    dy::F
    x::A
    y::A
    dx2::F
    dy2::F
end

"""
$(TYPEDSIGNATURES)

Builds a [`Grid`](@ref) with `nx * ny` cells over a domain of size `lx * ly`, deriving `dx`, `dy`, and the cell-center coordinate vectors `x`/`y`.
"""
function Grid(nx, ny, lx, ly)

    lx, ly = floattype(lx), floattype(ly)
    dx = lx/(nx-1)
    dy = ly/(ny-1)
    x = collect(0:dx:(nx-1)*dx)
    y = collect(0:dy:(ny-1)*dy)
    return Grid(nx, ny, lx, ly, dx, dy, x, y, dx^2, dy^2)

end
