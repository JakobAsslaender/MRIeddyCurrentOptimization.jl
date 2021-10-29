using MRIeddyCurrentOptimization
using BenchmarkTools
using LinearAlgebra
using Random

nFA = 571
nCyc = 872;

s, v = eigen([0 1 0; 0 0 1; 1 0 1])
GA1 = real(v[1,end] / v[end,end])

GA2 = real(v[2,end] / v[end,end])

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
kv = reshape(k, 3, nCyc*nFA);

N = 10_000_000

order = Int32.(1:(nCyc*nFA))
order = reshape(order, nFA, nCyc)

MRIeddyCurrentOptimization.cost(kv,order)

SimulatedAnneling!(kv, order, N, nFA, nCyc)

MRIeddyCurrentOptimization.cost(kv,order)

N = 1_000
@benchmark SimulatedAnneling!($kv, $order, $N, $nFA, $nCyc, rng = $(MersenneTwister(12345)))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

