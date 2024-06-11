"""
    cost(k, order; p=3, w_even=1)

Calculates the cost of the trajectory `k` when acquired in the order `order`.

# Required Arguments
- `k::Matrix{Number}`: 3 x (Nt Nc) matrix containing the [x,y,z] coordinates of the first data point of each spoke
- `order::Matrix{Int}`: Nt x Nc matrix containing the indices in which the spokes in `k` are acquired. 

# Optional Arguments
- `p::Number`: exponent to scale the squared Euclidean distance. Default is `p=3`, which is equivalent to `p=6` in the paper, as the Euclidean distance is already squared in the code. 
- `w_even::Number`: weighting factor of the evenly numbered jumps. The default is `w_even=1` which weights even and odd jumps equally. Set `w_even=0` to approximate [Bieri's pairing approach](https://doi.org/10.1002/mrm.20527). 

# Examples
```jldoctest
julia> using MRIeddyCurrentOptimization

julia> Nt  = 10;

julia> Nc = 5;

julia> θ = acos.(((0:(Nc * Nt - 1)) * 0.46557) .% 1);

julia> φ = Float64.(0:(Nc * Nt - 1)) * 2 * pi * 0.6823;

julia> k = zeros(3, length(θ));

julia> k[3,:] = cos.(θ);

julia> k[2,:] = sin.(θ) .* sin.(φ);

julia> k[1,:] = sin.(θ) .* cos.(φ);

julia> k = reshape(k, 3, Nc, Nt);

julia> k = permutedims(k, (1, 3, 2));

julia> k = reshape(k, 3, Nc*Nt)
3×50 Matrix{Float64}:
 1.0          -0.802397   0.334292  …  -0.762115  0.866723  -0.531067
 0.0           0.498671  -0.676983     -0.62806   0.116138   0.238983
 6.12323e-17   0.32785    0.6557        0.15723   0.48508    0.81293

julia> order = reshape(Int32.(1:(Nc*Nt)), Nt, Nc)
10×5 Matrix{Int32}:
  1  11  21  31  41
  2  12  22  32  42
  3  13  23  33  43
  4  14  24  34  44
  5  15  25  35  45
  6  16  26  36  46
  7  17  27  37  47
  8  18  28  38  48
  9  19  29  39  49
 10  20  30  40  50

julia> cost(k, order)
953.5601055273712

julia> cost(k, order; p=1, w_even=0)
63.2591698560959

```
"""
function cost(k, order; p=3, w_even=1)
    dk = k[:,order[1:end - 1]] - k[:,order[2:end]]
    dk = reduce(+, dk.^2, dims=1).^p
    F = sum(dk[1:2:end]) + w_even * sum(dk[2:2:end])  
    return F
end

"""
    delta_cost(k, order, t, c, c̃[; p=3, w_even=1])
    delta_cost(k, order, t, Nt, c, c̃, p, w_even)

Calculates the change in the cost when swapping the k-space spokes in the `c`ᵗʰ and `c̃`ᵗʰ cycles for the `t`ᵗʰ Tᵣ. 

# Required Arguments
- `k::Matrix{Number}`: 3 x (Nt Nc) matrix containing the [x,y,z] coordinates of the first data point of each spoke
- `order::Matrix{Int}`: Nt x Nc matrix containing the indices in which the spokes in `k` are acquired. 
- `t::Int`: index of the flip angle. Must be in the range [1, size(order,1)].
- `c::Int`: index of the first cycle. Must be in the range [1, size(order,2)].
- `c̃::Int`: index of the second cycle. Must be in the range [1, size(order,2)].

# Optional Arguments
- `p::Number`: exponent to scale the squared Euclidean distance. Default is `p=3`, which is equivalent to `p=6` in the paper, as the Euclidean distance is already squared in the code. 
- `w_even::Number`: weighting factor of the evenly numbered jumps. The default is `w_even=1` which weights even and odd jumps equally. Set `w_even=0` to approximate [Bieri's pairing approach](https://doi.org/10.1002/mrm.20527).

# Examples
```jldoctest
julia> using MRIeddyCurrentOptimization

julia> Nt  = 10;

julia> Nc = 5;

julia> θ = acos.(((0:(Nc * Nt - 1)) * 0.46557) .% 1);

julia> φ = Float64.(0:(Nc * Nt - 1)) * 2 * pi * 0.6823;

julia> k = zeros(3, length(θ));

julia> k[3,:] = cos.(θ);

julia> k[2,:] = sin.(θ) .* sin.(φ);

julia> k[1,:] = sin.(θ) .* cos.(φ);

julia> k = reshape(k, 3, Nc, Nt);

julia> k = permutedims(k, (1, 3, 2));

julia> k = reshape(k, 3, Nc*Nt)
3×50 Matrix{Float64}:
 1.0          -0.802397   0.334292  …  -0.762115  0.866723  -0.531067
 0.0           0.498671  -0.676983     -0.62806   0.116138   0.238983
 6.12323e-17   0.32785    0.6557        0.15723   0.48508    0.81293

julia> order = reshape(Int32.(1:(Nc*Nt)), Nt, Nc)
10×5 Matrix{Int32}:
  1  11  21  31  41
  2  12  22  32  42
  3  13  23  33  43
  4  14  24  34  44
  5  15  25  35  45
  6  16  26  36  46
  7  17  27  37  47
  8  18  28  38  48
  9  19  29  39  49
 10  20  30  40  50

julia> delta_cost(k, order, 7, 2, 4)
-110.52694915027755

julia> delta_cost(k, order, 7, 2, 4; p=5, w_even=0)
-1033.0264683547257

```
"""
function delta_cost(k, order, t, c, c̃; p=3, w_even=1)
    Nt, _ = size(order)
    delta_cost(k, order, t, Nt, c, c̃, p, w_even)
end

function delta_cost(k, order, t, Nt, c, c̃, p, w_even)
    w12 = t % 2 == 0 ? w_even : 1
    w23 = t % 2 != 0 ? w_even : 1

    i1 = (c - 1) * Nt + t
    i2 = (c̃ - 1) * Nt + t

    x11 = order[i1 - 1]
    x12 = order[i1]
    x13 = order[i1 + 1]
    x21 = order[i2 - 1]
    x22 = order[i2]
    x23 = order[i2 + 1]

    Δk  = (k[1,x11] - k[1,x12])^2
    Δk += (k[2,x11] - k[2,x12])^2
    Δk += (k[3,x11] - k[3,x12])^2
    ΔF = -w12 * Δk^p

    Δk  = (k[1,x11] - k[1,x22])^2
    Δk += (k[2,x11] - k[2,x22])^2
    Δk += (k[3,x11] - k[3,x22])^2
    ΔF += w12 * Δk^p


    Δk  = (k[1,x12] - k[1,x13])^2
    Δk += (k[2,x12] - k[2,x13])^2
    Δk += (k[3,x12] - k[3,x13])^2
    ΔF -= w23 * Δk^p

    Δk  = (k[1,x22] - k[1,x13])^2
    Δk += (k[2,x22] - k[2,x13])^2
    Δk += (k[3,x22] - k[3,x13])^2
    ΔF += w23 * Δk^p


    Δk  = (k[1,x21] - k[1,x22])^2
    Δk += (k[2,x21] - k[2,x22])^2
    Δk += (k[3,x21] - k[3,x22])^2
    ΔF -= w12 * Δk^p

    Δk  = (k[1,x21] - k[1,x12])^2
    Δk += (k[2,x21] - k[2,x12])^2
    Δk += (k[3,x21] - k[3,x12])^2
    ΔF += w12 * Δk^p


    Δk  = (k[1,x22] - k[1,x23])^2
    Δk += (k[2,x22] - k[2,x23])^2
    Δk += (k[3,x22] - k[3,x23])^2
    ΔF -= w23 * Δk^p

    Δk  = (k[1,x12] - k[1,x23])^2
    Δk += (k[2,x12] - k[2,x23])^2
    Δk += (k[3,x12] - k[3,x23])^2
    ΔF += w23 * Δk^p

    return ΔF
end


"""
    SimulatedAnneling!(k, order[;N_iter=1_000_000_000, p=3, w_even=1, rng = MersenneTwister(12345), verbose = false])

Performs the simulated annealing algorithm and writes the result in-place in `order`. For convenience, `order` is also returned.

# Required Arguments
- `k::Matrix{Number}`: 3 x (Nt Nc) matrix containing the [x,y,z] coordinates of the first data point of each spoke
- `order::Matrix{Int}`: Nt x Nc matrix containing the indices in which the spokes in `k` are acquired. This matrix is overwritten by the algorithm with the optimized index matrix. 

# Optional Arguments
- `N_iter::Int`: Number of iterations. The default is `1e9`
- `p::Number`: exponent to scale the squared Euclidean distance. Default is `p=3`, which is equivalent to `p=6` in the paper, as the Euclidean distance is already squared in the code. 
- `w_even::Number`: weighting factor of the evenly numbered jumps. The default is `w_even=1` which weights even and odd jumps equally. Set `w_even=0` to approximate [Bieri's pairing approach](https://doi.org/10.1002/mrm.20527).
- `rng`: seed for the random number generator. The default is `rng = MersenneTwister(12345)`. Use a different seed when repeating the algorithm for a different outcome. 
- `verbose::Boolean`: by default this flag is `false` and no output is printed. When set to `true`, the algorithm prints the cost at each full percent of the total number of iterations.

# Examples
```jldoctest
julia> using MRIeddyCurrentOptimization

julia> Nt  = 10;

julia> Nc = 5;

julia> θ = acos.(((0:(Nc * Nt - 1)) * 0.46557) .% 1);

julia> φ = Float64.(0:(Nc * Nt - 1)) * 2 * pi * 0.6823;

julia> k = zeros(3, length(θ));

julia> k[3,:] = cos.(θ);

julia> k[2,:] = sin.(θ) .* sin.(φ);

julia> k[1,:] = sin.(θ) .* cos.(φ);

julia> k = reshape(k, 3, Nc, Nt);

julia> k = permutedims(k, (1, 3, 2));

julia> k = reshape(k, 3, Nc*Nt)
3×50 Matrix{Float64}:
 1.0          -0.802397   0.334292  …  -0.762115  0.866723  -0.531067
 0.0           0.498671  -0.676983     -0.62806   0.116138   0.238983
 6.12323e-17   0.32785    0.6557        0.15723   0.48508    0.81293

julia> order = reshape(Int32.(1:(Nc*Nt)), Nt, Nc)
10×5 Matrix{Int32}:
  1  11  21  31  41
  2  12  22  32  42
  3  13  23  33  43
  4  14  24  34  44
  5  15  25  35  45
  6  16  26  36  46
  7  17  27  37  47
  8  18  28  38  48
  9  19  29  39  49
 10  20  30  40  50

julia> SimulatedAnneling!(k, order; N_iter=10)
10×5 Matrix{Int32}:
  1  11  21  31  41
  2  22  12  32  42
 13   3  43  33  23
  4  14  24  34  44
  5  15  25  35  45
 36  46  26   6  16
  7  17  27  37  47
  8  18  28  38  48
  9  19  29  39  49
 10  20  40  30  50

```
"""
function SimulatedAnneling!(k, order; N_iter=1_000_000_000, p=3, w_even=1, rng = MersenneTwister(12345), verbose = false)
    Nt, Nc = size(order)
    for ii = 1:N_iter
        t = Int32.(ceil.(rand(rng) * Nt))
        c = Int32.(ceil.(rand(rng) * Nc))
        c̃ = Int32.(ceil.(rand(rng) * Nc))

        if t == 1 && (c == 1 || c̃ == 1)
            t = 2
        elseif t == Nt && (c == Nc || c̃ == Nc)
            t = Nt - 1
        end

        ΔF = delta_cost(k, order, t, Nt, c, c̃, p, w_even)
        if exp(-ΔF / (1 - ii / N_iter)^6) > rand(rng)
            tmp = order[t,c̃]
            order[t,c̃] = order[t,c]
            order[t,c] = tmp
        end
        if verbose && ii % (N_iter / 100) == 0
            println(string(round(ii / N_iter * 1e2), "% completed; cost = ", cost(k, order, p=p, w_even=w_even)))
            flush(stdout)
        end
    end
    return order
end
