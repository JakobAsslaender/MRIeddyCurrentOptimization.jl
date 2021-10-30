using MRIeddyCurrentOptimization
using LinearAlgebra
using BenchmarkTools
using Test

# Define number of flip angles and cycles
N_t = 571
N_c = 872;

# calculate 2D golden means
s, v = eigen([0 1 0; 0 0 1; 1 0 1])
GA1 = real(v[1,end] / v[end,end])

# second one
GA2 = real(v[2,end] / v[end,end])

# set up 3D radial koosh ball trajectory
theta = acos.(((0:(N_c * N_t - 1)) * GA1) .% 1)
phi = Float64.(0:(N_c * N_t - 1)) * 2 * pi * GA2

k = zeros(3, length(theta))
k[3,:] = cos.(theta)
k[2,:] = sin.(theta) .* sin.(phi)
k[1,:] = sin.(theta) .* cos.(phi)

k = reshape(k, 3, N_c, N_t)
k = permutedims(k, (1, 3, 2))
k = reshape(k, 3, N_c * N_t);

# Set number of iterations
N = 10_000_000

# initalize with linear order
order = Int32.(1:(N_c * N_t))
order = reshape(order, N_t, N_c)

# choose values to change
t = 5
c = 10
c̃ = 20

# calculate delta cost
ΔF = delta_cost(k, order, t, c, c̃)

# calculate original cost
F0 = cost(k, order)

# change 
tmp = order[t,c̃]
order[t,c̃] = order[t,c]
order[t,c] = tmp

# calculate changed cost
F1 = cost(k, order)

# test
@test (F1 - F0) ≈ ΔF

# call simulated annealing algorithm that changes the order in place (as indicated by the ! at the end of the function call)
SimulatedAnneling!(k, order; N_iter=N)

## Test that the cost has decreased
F2 = cost(k, order)
@test F2 < F1

## benchmark 
print("Benchmark the call of delta_cost. This should be 0 allocations and around 22ns (on my 2015 i7):")
@btime delta_cost($k, $order, $t, $c, $c̃)

a = @benchmark delta_cost($k, $order, $t, $c, $c̃)
@test a.allocs == 0