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

function Grid(nx, ny, lx, ly)

    lx, ly = floattype(lx), floattype(ly)
    dx = lx/(nx-1)
    dy = ly/(ny-1)
    x = collect(0:dx:(nx-1)*dx)
    y = collect(0:dy:(ny-1)*dy)
    return Grid(nx, ny, lx, ly, dx, dy, x, y, dx^2, dy^2)

end
