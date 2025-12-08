# EtherArrays.jl

EtherArrays.jl is a wrapper of sized arrays on any device.

## Motivation

In GPU programming, coalesced memory access is the key point to performance. As a result, an SOA memory layout is required in most cases. With primitive arrays parsed to GPU lacking specified named-fields, a pointer-based container is needed on device side.

Supposing we have a column-majored array of physical simulation field called `data`:

```txt
m1, rho1, u1, v1, s1_11, s1_12, s1_21, s1_22...
m2, rho2, u2, v2, s2_11, s2_12, s2_21, s2_22...
m3, rho3, u3, v3, s3_11, s3_12, s3_21, s3_22...
.   .    .   .   .   .
.   .    .   .   .   .
mn, rhon, un, vn, sn_11, sn_12, sn_21, sn_22...
```

which means a physical element has: mass $m$, density $\rho$, velocity vector $\vec{u}=(u,v)$ and 2-ordered stress tensor $
\sigma=\tilde{s}_{ij}
$.

Largely thanks to [`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) and Julia type system, we can redefine container type in kernel function on device. 

`EtherArrays.jl` provides such a mutable container type `E2Array` that can be used in kernel functions.

```julia
using EtherArrays
using KernelAbstractions

data = ... # the device side array
const nt = (m=1, rho=2, u=3, sigma=5) # named tuple for field indices

@kernel function f!(data)
    idx = @index(Global)
    m = data[idx, nt.m]
    rho = data[idx, nt.rho]
    u = E2Vector{2}(idx, nt.u, data) # subtype of StaticVector
    sigma = E2Matrix{2, 2}(idx, nt.sigma, data) # a subtype of StaticMatrix
    ... # operate the container as `StaticArrays.MArray`
end
```

Other memory layouts can be extended by implementing the `EtherArray` interface like `E2Array <: EtherArray`.