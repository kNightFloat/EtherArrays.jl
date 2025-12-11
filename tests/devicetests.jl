#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2025-12-08 13:49:03
  @ license: MIT
  @ language: Julia
  @ declaration: EtherArrays.jl is a wrapper of sized arrays on any device.
  @ description: customizedtests.jl
 =#

# ! require gpu, here I test the package on a `oneAPI` GPU

using Test # require Test stdlib
using KernelAbstractions # require KernelAbstractions.jl package
using oneAPI # require oneAPI.jl package
using StaticArrays # require StaticArrays.jl package
using Pkg
Pkg.activate(joinpath(@__DIR__, "..")) # activate the package environment

using EtherArrays # * the tested package

const kTContainer = oneAPI.oneArray
const kBackend = oneAPI.oneAPIBackend()
const kTReal = Float32

@testset "oneapi kernel test" begin
    data_cpu = randn(kTReal, 3, 7)
    data_gpu = KernelAbstractions.zeros(kBackend, kTReal, size(data_cpu)...)
    Base.copyto!(data_gpu, data_cpu)

    @kernel function test_ker!(data)
        idx::Int = @index(Global)
        s = E2Scalar(idx, 1, data)
        s .+= 1
        v = E2Vector{2}(idx, 2, data)
        v .+= StaticArrays.norm(v)
        m = E2SquareMatrix{2}(idx, 4, data)
        m .= StaticArrays.inv(m)
        m2 = similar(m)
        m2 .+= 2
        m .= m2
    end

    test_ker!(kBackend, 3)(data_gpu, ndrange = (3,))
    KernelAbstractions.synchronize(kBackend)

    for i in 1:3
        data_cpu[i, 1] += 1
        v = data_cpu[i, 2:3]
        data_cpu[i, 2:3] .+= sqrt(v[1]^2 + v[2]^2)
        m = zeros(kTReal, 2, 2)
        m[1, 1] = data_cpu[i, 4]
        m[2, 1] = data_cpu[i, 5]
        m[1, 2] = data_cpu[i, 6]
        m[2, 2] = data_cpu[i, 7]
        m .= inv(m)
        m2 = similar(m)
        m2 .+= 2
        m .= m2
        data_cpu[i, 4:7] .= reshape(m, 4)
    end

    @test sum((Array(data_gpu) .- data_cpu) .^ 2) < 1e-6
end
