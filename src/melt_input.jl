abstract type AbstractMeltInput end

struct ConstantMeltInput <: AbstractMeltInput end

function initialize_ieb!(::ConstantMeltInput, state::State, ieb::AbstractArray)
    state.ieb .= ieb
end

function compute_ieb!(::ConstantMeltInput, ieb::AbstractArray, i::Int, j::Int)
    return ieb[i, j]
end
