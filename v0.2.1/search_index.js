var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"Authors = \"Jakob Assländer and Sebastian Flassbeck\"\nCurrentModule = MRIeddyCurrentOptimization","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"In the following, you find the documentation of all exported functions of the MRIeddyCurrentOptimization.jl package:","category":"page"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [MRIeddyCurrentOptimization]","category":"page"},{"location":"api/#MRIeddyCurrentOptimization.SimulatedAnneling!-Tuple{Any, Any}","page":"API","title":"MRIeddyCurrentOptimization.SimulatedAnneling!","text":"SimulatedAnneling!(k, order[;N_iter=1_000_000_000, p=3, w_even=1, rng = MersenneTwister(12345), verbose = false])\n\nPerforms the simulated annealing algorithm and writes the result in-place in order. For convenience, order is also returned.\n\nRequired Arguments\n\nk::Matrix{Number}: 3 x (Nt Nc) matrix containing the [x,y,z] coordinates of the first data point of each spoke\norder::Matrix{Int}: Nt x Nc matrix containing the indices in which the spokes in k are acquired. This matrix is overwritten by the algorithm with the optimized index matrix. \n\nOptional Arguments\n\nN_iter::Int: Number of iterations. The default is 1e9\np::Number: exponent to scale the squared Euclidean distance. Default is p=3, which is equivalent to p=6 in the paper, as the Euclidean distance is already squared in the code. \nw_even::Number: weighting factor of the evenly numbered jumps. The default is w_even=1 which weights even and odd jumps equally. Set w_even=0 to approximate Bieri's pairing approach.\nrng: seed for the random number generator. The default is rng = MersenneTwister(12345). Use a different seed when repeating the algorithm for a different outcome. \nverbose::Boolean: by default this flag is false and no output is printed. When set to true, the algorithm prints the cost at each full percent of the total number of iterations.\n\nExamples\n\njulia> using MRIeddyCurrentOptimization\n\njulia> Nt  = 10;\n\njulia> Nc = 5;\n\njulia> θ = acos.(((0:(Nc * Nt - 1)) * 0.46557) .% 1);\n\njulia> φ = Float64.(0:(Nc * Nt - 1)) * 2 * pi * 0.6823;\n\njulia> k = zeros(3, length(θ));\n\njulia> k[3,:] = cos.(θ);\n\njulia> k[2,:] = sin.(θ) .* sin.(φ);\n\njulia> k[1,:] = sin.(θ) .* cos.(φ);\n\njulia> k = reshape(k, 3, Nc, Nt);\n\njulia> k = permutedims(k, (1, 3, 2));\n\njulia> k = reshape(k, 3, Nc*Nt)\n3×50 Matrix{Float64}:\n 1.0          -0.802397   0.334292  …  -0.762115  0.866723  -0.531067\n 0.0           0.498671  -0.676983     -0.62806   0.116138   0.238983\n 6.12323e-17   0.32785    0.6557        0.15723   0.48508    0.81293\n\njulia> order = reshape(Int32.(1:(Nc*Nt)), Nt, Nc)\n10×5 Matrix{Int32}:\n  1  11  21  31  41\n  2  12  22  32  42\n  3  13  23  33  43\n  4  14  24  34  44\n  5  15  25  35  45\n  6  16  26  36  46\n  7  17  27  37  47\n  8  18  28  38  48\n  9  19  29  39  49\n 10  20  30  40  50\n\njulia> SimulatedAnneling!(k, order; N_iter=10)\n10×5 Matrix{Int32}:\n  1  11  21  31  41\n  2  22  12  32  42\n 13   3  43  33  23\n  4  14  24  34  44\n  5  15  25  35  45\n 36  46  26   6  16\n  7  17  27  37  47\n  8  18  28  38  48\n  9  19  29  39  49\n 10  20  40  30  50\n\n\n\n\n\n\n","category":"method"},{"location":"api/#MRIeddyCurrentOptimization.cost-Tuple{Any, Any}","page":"API","title":"MRIeddyCurrentOptimization.cost","text":"cost(k, order; p=3, w_even=1)\n\nCalculates the cost of the trajectory k when acquired in the order order.\n\nRequired Arguments\n\nk::Matrix{Number}: 3 x (Nt Nc) matrix containing the [x,y,z] coordinates of the first data point of each spoke\norder::Matrix{Int}: Nt x Nc matrix containing the indices in which the spokes in k are acquired. \n\nOptional Arguments\n\np::Number: exponent to scale the squared Euclidean distance. Default is p=3, which is equivalent to p=6 in the paper, as the Euclidean distance is already squared in the code. \nw_even::Number: weighting factor of the evenly numbered jumps. The default is w_even=1 which weights even and odd jumps equally. Set w_even=0 to approximate Bieri's pairing approach. \n\nExamples\n\njulia> using MRIeddyCurrentOptimization\n\njulia> Nt  = 10;\n\njulia> Nc = 5;\n\njulia> θ = acos.(((0:(Nc * Nt - 1)) * 0.46557) .% 1);\n\njulia> φ = Float64.(0:(Nc * Nt - 1)) * 2 * pi * 0.6823;\n\njulia> k = zeros(3, length(θ));\n\njulia> k[3,:] = cos.(θ);\n\njulia> k[2,:] = sin.(θ) .* sin.(φ);\n\njulia> k[1,:] = sin.(θ) .* cos.(φ);\n\njulia> k = reshape(k, 3, Nc, Nt);\n\njulia> k = permutedims(k, (1, 3, 2));\n\njulia> k = reshape(k, 3, Nc*Nt)\n3×50 Matrix{Float64}:\n 1.0          -0.802397   0.334292  …  -0.762115  0.866723  -0.531067\n 0.0           0.498671  -0.676983     -0.62806   0.116138   0.238983\n 6.12323e-17   0.32785    0.6557        0.15723   0.48508    0.81293\n\njulia> order = reshape(Int32.(1:(Nc*Nt)), Nt, Nc)\n10×5 Matrix{Int32}:\n  1  11  21  31  41\n  2  12  22  32  42\n  3  13  23  33  43\n  4  14  24  34  44\n  5  15  25  35  45\n  6  16  26  36  46\n  7  17  27  37  47\n  8  18  28  38  48\n  9  19  29  39  49\n 10  20  30  40  50\n\njulia> cost(k, order)\n953.5601055273712\n\njulia> cost(k, order; p=1, w_even=0)\n63.2591698560959\n\n\n\n\n\n\n","category":"method"},{"location":"api/#MRIeddyCurrentOptimization.delta_cost-NTuple{5, Any}","page":"API","title":"MRIeddyCurrentOptimization.delta_cost","text":"delta_cost(k, order, t, c, c̃[; p=3, w_even=1])\ndelta_cost(k, order, t, Nt, c, c̃, p, w_even)\n\nCalculates the change in the cost when swapping the k-space spokes in the cᵗʰ and c̃ᵗʰ cycles for the tᵗʰ Tᵣ. \n\nRequired Arguments\n\nk::Matrix{Number}: 3 x (Nt Nc) matrix containing the [x,y,z] coordinates of the first data point of each spoke\norder::Matrix{Int}: Nt x Nc matrix containing the indices in which the spokes in k are acquired. \nt::Int: index of the flip angle. Must be in the range [1, size(order,1)].\nc::Int: index of the first cycle. Must be in the range [1, size(order,2)].\nc̃::Int: index of the second cycle. Must be in the range [1, size(order,2)].\n\nOptional Arguments\n\np::Number: exponent to scale the squared Euclidean distance. Default is p=3, which is equivalent to p=6 in the paper, as the Euclidean distance is already squared in the code. \nw_even::Number: weighting factor of the evenly numbered jumps. The default is w_even=1 which weights even and odd jumps equally. Set w_even=0 to approximate Bieri's pairing approach.\n\nExamples\n\njulia> using MRIeddyCurrentOptimization\n\njulia> Nt  = 10;\n\njulia> Nc = 5;\n\njulia> θ = acos.(((0:(Nc * Nt - 1)) * 0.46557) .% 1);\n\njulia> φ = Float64.(0:(Nc * Nt - 1)) * 2 * pi * 0.6823;\n\njulia> k = zeros(3, length(θ));\n\njulia> k[3,:] = cos.(θ);\n\njulia> k[2,:] = sin.(θ) .* sin.(φ);\n\njulia> k[1,:] = sin.(θ) .* cos.(φ);\n\njulia> k = reshape(k, 3, Nc, Nt);\n\njulia> k = permutedims(k, (1, 3, 2));\n\njulia> k = reshape(k, 3, Nc*Nt)\n3×50 Matrix{Float64}:\n 1.0          -0.802397   0.334292  …  -0.762115  0.866723  -0.531067\n 0.0           0.498671  -0.676983     -0.62806   0.116138   0.238983\n 6.12323e-17   0.32785    0.6557        0.15723   0.48508    0.81293\n\njulia> order = reshape(Int32.(1:(Nc*Nt)), Nt, Nc)\n10×5 Matrix{Int32}:\n  1  11  21  31  41\n  2  12  22  32  42\n  3  13  23  33  43\n  4  14  24  34  44\n  5  15  25  35  45\n  6  16  26  36  46\n  7  17  27  37  47\n  8  18  28  38  48\n  9  19  29  39  49\n 10  20  30  40  50\n\njulia> delta_cost(k, order, 7, 2, 4)\n-110.52694915027755\n\njulia> delta_cost(k, order, 7, 2, 4; p=5, w_even=0)\n-1033.0264683547257\n\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/JakobAsslaender/MRIeddyCurrentOptimization.jl/blob/master/docs/src/index.jl\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = MRIeddyCurrentOptimization","category":"page"},{"location":"#MRIeddyCurrentOptimization.jl","page":"Home","title":"MRIeddyCurrentOptimization.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for the MRIeddyCurrentOptimization.jl package, which implements a simulated annealing algorithm to re-order k-space lines for minimal eddy current artifacts. The approach is describe in detail in the corresponding paper. In the following, we give a brief tutorial. The documentation of all exported functions can be found in the API Section.","category":"page"},{"location":"#Tutorial","page":"Home","title":"Tutorial","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"The main function is SimulatedAnneling!(k, order), which, for a given k-space trajectory k, optimizes the index matrix order. The function randomly chooses a timepoint t of the spin dynamics and, within t, attempts to swap the two random cylce indices c and c̃. If the swap is beneficial, the indices are swapped. In line with the simulated annealing theory, even unbeneficial swaps are performed with a certain probabilty that is comparably high in the first iterations and goes to zero for later iterations. In the following, we explain the interface of the package at the example of the optimization used in the paper where we optimize a 3D radial koosh-ball trajectry.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For this example, we need the following packages:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using MRIeddyCurrentOptimization\nusing BenchmarkTools\nusing LinearAlgebra\nusing Random\nusing Plots\nplotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native); # hide\nnothing #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"We define number of time points in our spin dynamics (i.e. the number of RF-pulses)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Nt = 981;\nnothing #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"as well as the number of cycles that we want to acquire","category":"page"},{"location":"","page":"Home","title":"Home","text":"Nc = 94;\nnothing #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"Like in the paper, we use the 2D golden means trajectory proposed by Chan et al., which can be calculated with an eigendecomposition. The first golden mean is","category":"page"},{"location":"","page":"Home","title":"Home","text":"s, v = eigen([0 1 0; 0 0 1; 1 0 1])\nϕ₁ = real(v[1,end] / v[end])","category":"page"},{"location":"","page":"Home","title":"Home","text":"and the second one is","category":"page"},{"location":"","page":"Home","title":"Home","text":"ϕ₂ = real(v[2,end] / v[end])","category":"page"},{"location":"","page":"Home","title":"Home","text":"With the golden means we calcualte the angles of the k-space spokes:","category":"page"},{"location":"","page":"Home","title":"Home","text":"θ = acos.(((0:(Nc * Nt - 1)) * ϕ₁) .% 1)\nφ = (0:(Nc * Nt - 1)) * 2π * ϕ₂;\nnothing #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"and calculate the k-space trajectory:","category":"page"},{"location":"","page":"Home","title":"Home","text":"k = zeros(3, length(θ))\nk[3,:] = cos.(θ)\nk[2,:] = sin.(θ) .* sin.(φ)\nk[1,:] = sin.(θ) .* cos.(φ);\nnothing #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"As discussed in the paper, a near-optimal k-space coverage is achieved by binning the first Nc angles into the first time point t₁, the next Nc angles into the second time point t₂ and so forth. Hence, we need to permute the dimensions of the k-space trajectory to re-order the spokes:","category":"page"},{"location":"","page":"Home","title":"Home","text":"k = reshape(k, 3, Nc, Nt)\nk = permutedims(k, (1, 3, 2))\nk = reshape(k, 3, Nc*Nt);\nnothing #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"Here, we initialize the simulated annealing algorithm with the default, i.e., with a linear ordering scheme:","category":"page"},{"location":"","page":"Home","title":"Home","text":"order = Int32.(1:(Nc*Nt))\norder = reshape(order, Nt, Nc)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The cost of this order, with the package's default cost function (p=3, which is equivalent to p=6 in the paper due to an additional squaring, and with w_even=1, i.e., with equal weights on even and odd jumps) is given by","category":"page"},{"location":"","page":"Home","title":"Home","text":"cost(k,order)","category":"page"},{"location":"","page":"Home","title":"Home","text":"We can visualize inital cost by plotting a histogram of the Euclidean distances:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Δk = k[:,order[1:end - 1]] - k[:,order[2:end]]\nΔk = vec(reduce(+, Δk.^2, dims=1))\np = histogram(Δk, bins=(0:0.01:1.5), xlabel=\"Euclidean distance\", ylabel=\"Number of occurrencess\", label = \"default ordering\")\nMain.HTMLPlot(p) #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"Like in the paper, we use a fixed 1 billion iterations:","category":"page"},{"location":"","page":"Home","title":"Home","text":"N = 1_000_000_000;\nnothing #hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"and we call the simulated annealing algorithm, which changes the matrix order in-place (as indicated by the ! at the end of the function call)","category":"page"},{"location":"","page":"Home","title":"Home","text":"SimulatedAnneling!(k, order, N_iter=N)","category":"page"},{"location":"","page":"Home","title":"Home","text":"One can appreciate the changed indices in order and we use the same cost function to compute the final cost that is substantially reduced:","category":"page"},{"location":"","page":"Home","title":"Home","text":"cost(k,order)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The redueced cost is also reflected in the histogram of the spherical distance:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Δk = k[:,order[1:end - 1]] - k[:,order[2:end]]\nΔk = vec(reduce(+, Δk.^2, dims=1))\nhistogram!(p, Δk, bins=(0:0.01:1.5), label = \"Uniform weighting\")\nMain.HTMLPlot(p) #hide","category":"page"},{"location":"#Benchmarking","page":"Home","title":"Benchmarking","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Last but not least, we can benchmark the code and verify that the code is non-allocating:","category":"page"},{"location":"","page":"Home","title":"Home","text":"N = 1_000\n@benchmark SimulatedAnneling!($k, $order, N_iter=$N, rng = $(MersenneTwister(12345)))","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"This page was generated using Literate.jl.","category":"page"}]
}
