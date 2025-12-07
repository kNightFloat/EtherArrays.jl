#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2025-12-07 21:43:53
  @ license: MIT
  @ language: Julia
  @ declaration: EtherArrays.jl is a wrapper of sized arrays on any device.
  @ description: /
 =#

using Test
using StaticArrays
using EtherArrays

@testset "types 1" begin
    @test EtherScalar{Float64} === EtherArray{Tuple{}, Float64, 0}
    @test EtherVector{3, Int32} === EtherArray{Tuple{3}, Int32, 1}
    @test EtherMatrix{2, 4, Float32} === EtherArray{Tuple{2, 4}, Float32, 2}
    @test EtherSquareMatrix{5, Int64} === EtherArray{Tuple{5, 5}, Int64, 2}

    _T = Float32
    data = randn(_T, 3, 8)
    @test E2Scalar(1, 1, data) isa E2Array{Tuple{}, _T, 0}
    @test E2Vector{3}(1, 1, data) isa E2Array{Tuple{3}, _T, 1}
    @test E2Matrix{2, 4}(1, 1, data) isa E2Array{Tuple{2, 4}, _T, 2}
    @test E2SquareMatrix{2}(1, 1, data) isa E2Array{Tuple{2, 2}, _T, 2}
end

@testset "types 2" begin
    _T = Float32
    data = randn(_T, 3, 8)
    e2scalar = E2Scalar(1, 1, data)
    e2vector = E2Vector{3}(1, 1, data)
    e2matrix = E2Matrix{2, 4}(1, 1, data)
    e2squarematrix = E2SquareMatrix{2}(1, 1, data)

    @test size(e2scalar) == ()
    @test size(e2vector) == (3,)
    @test size(e2matrix) == (2, 4)
    @test size(e2squarematrix) == (2, 2)
end
