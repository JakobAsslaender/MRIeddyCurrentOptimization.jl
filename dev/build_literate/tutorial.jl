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
phi = (0:(nCyc * nFA - 1)) * 2 * pi * GA2

k = zeros(3, length(theta))
k[3,:] = cos.(theta)
k[2,:] = sin.(theta) .* sin.(phi)
k[1,:] = sin.(theta) .* cos.(phi)

k = reshape(k, 3, nCyc, nFA)
k = permutedims(k, (1, 3, 2))
k = reshape(k, 3, nCyc*nFA);

N = 10_000_000

order = Int32.(1:(nCyc*nFA))
order = reshape(order, nFA, nCyc)

cost(k,order)

SimulatedAnneling!(k, order, N_iter=N)

cost(k,order)

N = 1_000
@benchmark SimulatedAnneling!($k, $order, N_iter=$N, rng = $(MersenneTwister(12345)))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

