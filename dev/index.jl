#md # ```@meta
#md # CurrentModule = MRIeddyCurrentOptimization
#md # ```

#md # # MRIeddyCurrentOptimization.jl

#md # Documentation for the [MRIeddyCurrentOptimization.jl](https://github.com/JakobAsslaender/MRIeddyCurrentOptimization.jl) package, which implements a simulated annealing algorithm to re-order k-space lines for minimal eddy current artifacts. The approach is describe in detail in the corresponding [paper](https://TODO.url). In the following, we give a brief tutorial. The documentation of all exported functions can be found in the [API](@ref) Section. 

# ## Tutorial
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/index.ipynb)

# The main function is [`SimulatedAnneling!(k, order)`](@ref), which, for a given k-space trajectory `k`, optimizes the index matrix `order`. The function randomly chooses a timepoint `t` of the spin dynamics and, within `t`, attempts to swap the two random cylce indices `c` and `c̃`. If the swap is beneficial, the indices are swapped. In line with the simulated annealing theory, even unbeneficial swaps are performed with a certain probabilty that is comparably high in the first iterations and goes to zero for later iterations. In the following, we explain the interface of the package at the example of the optimization used in the paper where we optimize a 3D radial koosh-ball trajectry. 

# For this example, we need the following packages:
using MRIeddyCurrentOptimization
using BenchmarkTools
using LinearAlgebra
using Random
using Plots
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native); # hide #md

# We define number of time points in our spin dynamics (i.e. the number of RF-pulses)
Nt = 981;

# as well as the number of cycles that we want to acquire
Nc = 94;

# Like in the paper, we use the 2D golden means trajectory proposed by [Chan et al.](https://doi.org/10.1002/mrm.21837), which can be calculated with an eigendecomposition. The first golden mean is
s, v = eigen([0 1 0; 0 0 1; 1 0 1])
ϕ₁ = real(v[1,end] / v[end])

# and the second one is 
ϕ₂ = real(v[2,end] / v[end])

# With the golden means we calcualte the angles of the k-space spokes:
θ = acos.(((0:(Nc * Nt - 1)) * ϕ₁) .% 1)
φ = (0:(Nc * Nt - 1)) * 2π * ϕ₂;

# and calculate the k-space trajectory:
k = zeros(3, length(θ))
k[3,:] = cos.(θ)
k[2,:] = sin.(θ) .* sin.(φ)
k[1,:] = sin.(θ) .* cos.(φ);

# As discussed in the paper, a near-optimal k-space coverage is achieved by binning the first `Nc` angles into the first time point `t₁`, the next `Nc` angles into the second time point `t₂` and so forth. Hence, we need to permute the dimensions of the k-space trajectory to re-order the spokes:
k = reshape(k, 3, Nc, Nt)
k = permutedims(k, (1, 3, 2))
k = reshape(k, 3, Nc*Nt);

# Here, we initialize the simulated annealing algorithm with the *default*, i.e., with a linear ordering scheme:
order = Int32.(1:(Nc*Nt))
order = reshape(order, Nt, Nc)

# The cost of this order, with the package's default cost function (`p=3`, which is equivalent to `p=6` in the paper due to an additional squaring, and with `w_even=1`, i.e., with equal weights on even and odd jumps) is given by
cost(k,order)

# We can visualize inital cost by plotting a histogram of the Euclidean distances:
Δk = k[:,order[1:end - 1]] - k[:,order[2:end]]
Δk = vec(reduce(+, Δk.^2, dims=1))
p = histogram(Δk, bins=(0:0.01:1.5), xlabel="Euclidean distance", ylabel="Number of occurrencess", label = "default ordering")
#md Main.HTMLPlot(p) #hide

# Like in the paper, we use a fixed 1 billion iterations:
N = 1_000_000_000;

# and we call the simulated annealing algorithm, which changes the matrix `order` in-place (as indicated by the `!` at the end of the function call)
SimulatedAnneling!(k, order, N_iter=N)

# One can appreciate the changed indices in `order` and we use the same cost function to compute the final cost that is substantially reduced:
cost(k,order)

# The redueced cost is also reflected in the histogram of the spherical distance:
Δk = k[:,order[1:end - 1]] - k[:,order[2:end]]
Δk = vec(reduce(+, Δk.^2, dims=1))
histogram!(p, Δk, bins=(0:0.01:1.5), label = "Uniform weighting")
#md Main.HTMLPlot(p) #hide

# ## Benchmarking
# Last but not least, we can benchmark the code and verify that the code is non-allocating:
N = 1_000
@benchmark SimulatedAnneling!($k, $order, N_iter=$N, rng = $(MersenneTwister(12345)))