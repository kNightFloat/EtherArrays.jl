#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2025-12-08 18:09:53
  @ license: MIT
  @ language: Julia
  @ declaration: EtherArrays.jl is a wrapper of sized arrays on any device.
  @ description: /
 =#

using Test
using StaticArrays
using Pkg
Pkg.activate(joinpath(@__DIR__, "..")) # activate the package environment

using EtherArrays # * the tested package

@testset "mutable or not" begin
    data = randn(Float32, 3, 7)
    s = E2Scalar(1, 1, data)
    v = E2Vector{2}(2, 2, data)
    m = E2SquareMatrix{2}(3, 1, data)
    m2 = similar(m)

    @test m2 isa StaticArrays.MMatrix{2, 2, Float32, 4}
end
