using MRIeddyCurrentOptimization
using BenchmarkTools
using LinearAlgebra
using Random
using Plots

Nt = 975;

Nc = 94;

s, v = eigen([0 1 0; 0 0 1; 1 0 1])
ϕ₁ = real(v[1,end] / v[end])

ϕ₂ = real(v[2,end] / v[end])

θ = acos.(((0:(Nc * Nt - 1)) * ϕ₁) .% 1)
φ = (0:(Nc * Nt - 1)) * 2π * ϕ₂;

k = zeros(3, length(θ))
k[3,:] = cos.(θ)
k[2,:] = sin.(θ) .* sin.(φ)
k[1,:] = sin.(θ) .* cos.(φ);

k = reshape(k, 3, Nc, Nt)
k = permutedims(k, (1, 3, 2))
k = reshape(k, 3, Nc*Nt);

order = Int32.(1:(Nc*Nt))
order = reshape(order, Nt, Nc)

cost(k,order)

Δk = k[:,order[1:end - 1]] - k[:,order[2:end]]
Δk = vec(reduce(+, Δk.^2, dims=1))
p = histogram(Δk, bins=(0:0.01:1.5), xlabel="Euclidean distance on the unity sphere", ylabel="Number of occurrencess", label = "default ordering")

N = 1_000_000_000;

SimulatedAnneling!(k, order, N_iter=N)

cost(k,order)

Δk = k[:,order[1:end - 1]] - k[:,order[2:end]]
Δk = vec(reduce(+, Δk.^2, dims=1))
histogram!(p, Δk, bins=(0:0.01:1.5), label = "optimized ordering")

N = 1_000
@benchmark SimulatedAnneling!($k, $order, N_iter=$N, rng = $(MersenneTwister(12345)))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

