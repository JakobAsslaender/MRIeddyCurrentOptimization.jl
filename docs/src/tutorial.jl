#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/tutorial.ipynb)

# # Tutorial

# The core of generalized Bloch model is implemented in the function [`apply_hamiltonian_gbloch!`](@ref), which calculates the derivative `∂m/∂t` for a given magnetization vector `m` and stores it in-place in the the variable `∂m∂t`. The function interface is written in a way that we can directly feed it into a differential equation solver of the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package.

# For this example, we need the following packages:
using MRIeddyCurrentOptimization
using BenchmarkTools
using LinearAlgebra
using Random

# Define number of flip angles and cycles
N_t = 974
N_c = 94;

# calculate 2D golden means
s, v = eigen([0 1 0; 0 0 1; 1 0 1])
GA1 = real(v[1,end] / v[end,end])

# second one
GA2 = real(v[2,end] / v[end,end])

# set up 3D radial koosh ball trajectory
theta = acos.(((0:(N_c * N_t - 1)) * GA1) .% 1)
phi = (0:(N_c * N_t - 1)) * 2 * pi * GA2

k = zeros(3, length(theta))
k[3,:] = cos.(theta)
k[2,:] = sin.(theta) .* sin.(phi)
k[1,:] = sin.(theta) .* cos.(phi)

k = reshape(k, 3, N_c, N_t)
k = permutedims(k, (1, 3, 2))
k = reshape(k, 3, N_c*N_t);

# Set number of iterations
N = 10_000_000

# initalize with linear order
order = Int32.(1:(N_c*N_t))
order = reshape(order, N_t, N_c)

# calculate initial cost
cost(k,order)

# call simulated annealing algorithm that changes the order in place (as indicated by the ! at the end of the function call)
SimulatedAnneling!(k, order, N_iter=N)

# calculate the final cost
cost(k,order)

# # Benchmarking
N = 1_000
@benchmark SimulatedAnneling!($k, $order, N_iter=$N, rng = $(MersenneTwister(12345)))