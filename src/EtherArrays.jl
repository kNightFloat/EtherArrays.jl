#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2025-12-07 19:14:34
  @ license: MIT
  @ language: Julia
  @ declaration: EtherArrays.jl is a wrapper of sized arrays on any device.
  @ description: main file
 =#

module EtherArrays

export EtherArray
export EtherScalar, EtherVector, EtherMatrix, EtherSquareMatrix, EtherVecOrMat
export E2Array
export E2Scalar, E2Vector, E2Matrix, E2SquareMatrix, E2VecOrMat

import StaticArrays
import StaticArraysCore
import StaticArraysCore: tuple_prod
import StaticArraysCore: Size

import EtherMaths: unsafe_real

# * ===== ===== ===== ===== EtherArray ===== ===== ===== ===== * #

abstract type EtherArray{S, T <: Real, P} <: StaticArraysCore.StaticArray{S, T, P} end

const EtherScalar{T <: Real} = EtherArray{Tuple{}, T, 0}
const EtherVector{N, T <: Real} = EtherArray{Tuple{N}, T, 1}
const EtherMatrix{M, N, T <: Real} = EtherArray{Tuple{M, N}, T, 2}
const EtherSquareMatrix{N, T <: Real} = EtherArray{Tuple{N, N}, T, 2}
const EtherVecOrMat{T} = Union{EtherVector{<:Any, T}, EtherMatrix{<:Any, <:Any, T}}

@inline function Base.Tuple(a::EtherArray{S, T, P})::NTuple{tuple_prod(S), T} where {S <: Tuple, T <: Real, P}
    L = tuple_prod(S)
    return ntuple(i -> getindex(a, i), L)
end

@inline function Base.strides(a::EtherArray{S, T, P}) where {S <: Tuple, T <: Real, P}
    return Base.size_to_strides(1, size(a)...)
end

@inline function StaticArrays.similar_type(::Type{SA}, ::Type{T}, s::Size{S}) where {SA <: EtherArray, T, S}
    return StaticArrays.mutable_similar_type(T, s, StaticArrays.length_val(s))
end

# * ===== ===== E2Array ===== ===== * #

#= NOTE:
    E2Array means a wrapper of 2D arrays on device.
    As in julia, data is stored in column-major order.
    For memory coalescing on GPUs, SOA requires data in a specified struct aligned along row.
    For data post processing on CPUs, the 2D data structure is more friendly.
    Thus, E2Array is designed to store 2D arrays in row-major order.
=#
struct E2Array{S, T <: Real, P} <: EtherArray{S, T, P}
    row_::Int
    col_::Int
    data_::Ref
end

const E2Scalar{T <: Real} = E2Array{Tuple{}, T, 0}
const E2Vector{N, T <: Real} = E2Array{Tuple{N}, T, 1}
const E2Matrix{M, N, T <: Real} = E2Array{Tuple{M, N}, T, 2}
const E2SquareMatrix{N, T <: Real} = E2Array{Tuple{N, N}, T, 2}
const E2VecOrMat{T} = Union{E2Vector{<:Any, T}, E2Matrix{<:Any, <:Any, T}}

@inline function _row(a::E2Array{S, T, P})::Int where {S <: Tuple, T <: Real, P}
    return getfield(a, :row_)
end

@inline function _col(a::E2Array{S, T, P})::Int where {S <: Tuple, T <: Real, P}
    return getfield(a, :col_)
end

@inline function _data(a::E2Array{S, T, P})::Ref where {S <: Tuple, T <: Real, P}
    return getfield(a, :data_).x
end

@inline function Base.getindex(a::E2Array{S, T, P}, i::Int) where {S <: Tuple, T <: Real, P}
    # ! key methods for StaticArrays
    return @inbounds _data(a)[_row(a), _col(a) - 1 + i]
end

@inline function Base.setindex!(a::E2Array{S, T, P}, v::Real, i::Int) where {S <: Tuple, T <: Real, P}
    # ! key methods for StaticArrays
    return @inbounds _data(a)[_row(a), _col(a) - 1 + i] = unsafe_real(T, v)
end

# * ===== Constructors ===== * #

@inline function E2Array{S, T, P}(
    row::Integer,
    col::Integer,
    data::AbstractArray{T, 2},
)::E2Array{S, T, P} where {S <: Tuple, T <: Real, P}
    return E2Array{S, T, P}(Int(row), Int(col), Ref{typeof(data)}(data))
end

@inline function E2Array{S}(
    row::Integer,
    col::Integer,
    data::AbstractArray{T, 2},
)::E2Array{S, eltype(data), length(S.parameters)} where {S <: Tuple, T <: Real}
    return E2Array{S, eltype(data), length(S.parameters)}(Int(row), Int(col), Ref{typeof(data)}(data))
end

@inline function E2Array{S, T}(
    row::Integer,
    col::Integer,
    data::AbstractArray{T, 2},
)::E2Array{S, T, length(S.parameters)} where {S <: Tuple, T <: Real}
    return E2Array{S, T, length(S.parameters)}(Int(row), Int(col), Ref{typeof(data)}(data))
end

@inline function E2Array{S, T, P}()::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> zero(T), tuple_prod(S)))
end

@inline function E2Array{S, T, P}(
    x::NTuple{L, T},
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P, L}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> x[i], tuple_prod(S)))
end

@inline function E2Array{S, T, P}(
    ::UndefInitializer,
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(ntuple(i -> zero(T), tuple_prod(S)))
end

@inline function E2Array{S, T, P}(
    x::Base.Tuple,
)::StaticArraysCore.MArray{S, T, P, tuple_prod(S)} where {S <: Tuple, T <: Real, P}
    return StaticArrays.MArray{S, T, P, tuple_prod(S)}(x)
end

# * ===== Constructors for S, V, M ===== * #

@inline function E2Scalar(row::Integer, col::Integer, data::AbstractArray{T, 2}) where {T <: Real}
    return E2Scalar{T}(row, col, data)
end

@inline function E2Vector{N}(row::Integer, col::Integer, data::AbstractArray{T, 2}) where {N, T <: Real}
    return E2Vector{N, T}(row, col, data)
end

@inline function E2Matrix{M, N}(row::Integer, col::Integer, data::AbstractArray{T, 2}) where {M, N, T}
    return E2Matrix{M, N, T}(row, col, data)
end

@inline function E2SquareMatrix{N}(row::Integer, col::Integer, data::AbstractArray{T, 2}) where {N, T <: Real}
    return E2SquareMatrix{N, T}(row, col, data)
end

end # module EtherArrays
