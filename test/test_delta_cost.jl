using MRIeddyCurrentOptimization
using MRIeddyCurrentOptimization: cost, delta_cost
using LinearAlgebra
using BenchmarkTools
using Test

# Define number of flip angles and cycles
nFA = 571
nCyc = 872;

# calculate 2D golden means
s, v = eigen([0 1 0; 0 0 1; 1 0 1])
GA1 = real(v[1,end] / v[end,end])

# second one
GA2 = real(v[2,end] / v[end,end])

# set up 3D radial koosh ball trajectory
theta = acos.(((0:(nCyc * nFA - 1)) * GA1) .% 1)
phi = Float64.(0:(nCyc * nFA - 1)) * 2 * pi * GA2

theta = reshape(theta, nCyc, nFA)
phi   = reshape(phi, nCyc, nFA)
theta = vec(theta)
phi   = vec(phi)

k = zeros(3, length(theta))
k[3,:] = cos.(theta)
k[2,:] = sin.(theta) .* sin.(phi)
k[1,:] = sin.(theta) .* cos.(phi)

k = reshape(k, 3, nCyc, nFA)
k = permutedims(k, (1, 3, 2))
kv = reshape(k, 3, nCyc * nFA);

# Set number of iterations
N = 10_000_000

# initalize with linear order
order = Int32.(1:(nCyc * nFA))
order = reshape(order, nFA, nCyc)

# choose values to change
iFA = 5
iC1 = 10
iC2 = 20

# calculate delta cost
Δc = delta_cost(kv, order, iFA, nFA, iC1, iC2)

# calculate original cost
c0 = cost(kv, order)

# change 
tmp = order[iFA,iC2]
order[iFA,iC2] = order[iFA,iC1]
order[iFA,iC1] = tmp

# calculate changed cost
c1 = cost(kv, order)

# test
@test (c1 - c0) ≈ Δc

# call simulated annealing algorithm that changes the order in place (as indicated by the ! at the end of the function call)
SimulatedAnneling!(kv, order, N, nFA, nCyc)

## Test that the cost has decreased
c2 = cost(kv, order)
@test c2 < c1

## benchmark 
print("Benchmark the call of delta_cost. This should be 0 allocations and around 22ns (on my 2015 i7):")
@btime delta_cost($kv, $order, $iFA, $nFA, $iC1, $iC2)

a = @benchmark delta_cost($kv, $order, $iFA, $nFA, $iC1, $iC2)
@test a.allocs == 0