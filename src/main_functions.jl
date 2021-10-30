
function cost(k, x; p=3, w_even=1)
    xv = vec(x)
    dk = k[:,xv[1:end - 1]] - k[:,x[2:end]]
    dk = reduce(+, dk.^2, dims=1).^p
    F = sum(dk[1:2:end]) + w_even * sum(dk[2:2:end])  
    return F
end

function delta_cost(k, order, iFA, iC1, iC2; p=3, w_even=1)
    nFA, _ = size(order)
    delta_cost(k, order, iFA, nFA, iC1, iC2, p, w_even)
end

function delta_cost(k, order, iFA, nFA, iC1, iC2, p, w_even)
    w12 = iFA % 2 == 0 ? w_even : 1
    w23 = iFA % 2 != 0 ? w_even : 1

    i1 = (iC1 - 1) * nFA + iFA
    i2 = (iC2 - 1) * nFA + iFA

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

Performs the simulated annealing algorith and writes the result in-place in `order` and returns `order` for convenience.

# Arguemnts
- `k::Matrix{Number}`: 3 x N_spokes matrix containing the [x,y,z] coordinates of the first data point of each spoke
- `order::Matrix{Int}`: nFA x nCyc matrix containing the indices in which the spokes in `k` are acquired. This matrix is overwritten by the algorithm with the optimized index matrix. 


Optional:
- `N_iter::Int`: Number of iterations. The default is `1e9`
- `T2s_max::Number`: upper bound of the `T2s` range in seconds
- `p::Number`: exponent to scale the squared-Euclidean distance. Default is `p=3`, which is equivalent to `p=6` in the paper, as the Euclidean distance is already squared. 
- `w_even::Number`: weighting factor of the even jumps. The default is `w_even=1` which weights even and odd equally. Choose `w_even=0` for Bieri's paring approch. 
- `rng`: seed for the random number generator. The default is `rng = MersenneTwister(12345)`. Use a different seed when repeating the algorithm for a different outcome. 
- `verbose::Boolean`: by default this flag is `false` and no output is printed. When set to `true`, the algorithm prints the cost at each full percent of runtime.

# Examples
```jldoctest
julia> using MRIeddyCurrentOptimization

julia> nFA  = 10;

julia> nCyc = 5;

julia> theta = acos.(((0:(nCyc * nFA - 1)) * 0.46557) .% 1);

julia> phi = Float64.(0:(nCyc * nFA - 1)) * 2 * pi * 0.6823;

julia> k = zeros(3, length(theta));

julia> k[3,:] = cos.(theta);

julia> k[2,:] = sin.(theta) .* sin.(phi);

julia> k[1,:] = sin.(theta) .* cos.(phi);

julia> k = reshape(k, 3, nCyc, nFA);

julia> k = permutedims(k, (1, 3, 2));

julia> k = reshape(k, 3, nCyc*nFA)
3×50 Matrix{Float64}:
 1.0          -0.802397   0.334292  …  -0.762115  0.866723  -0.531067
 0.0           0.498671  -0.676983     -0.62806   0.116138   0.238983
 6.12323e-17   0.32785    0.6557        0.15723   0.48508    0.81293

julia> order = reshape(Int32.(1:(nCyc*nFA)), nFA, nCyc)
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
    nFA, N_c = size(order)
    for ii = 1:N_iter
        iFA = Int32.(ceil.(rand(rng) * nFA))
        iC1 = Int32.(ceil.(rand(rng) * N_c))
        iC2 = Int32.(ceil.(rand(rng) * N_c))

        if iFA == 1 && (iC1 == 1 || iC2 == 1)
            iFA = 2
        elseif iFA == nFA && (iC1 == N_c || iC2 == N_c)
            iFA = nFA - 1
        end

        c = delta_cost(k, order, iFA, nFA, iC1, iC2, p, w_even)
        if exp(-c / (1 - ii / N_iter)^6) > rand(rng)
            tmp = order[iFA,iC2]
            order[iFA,iC2] = order[iFA,iC1]
            order[iFA,iC1] = tmp
        end
        if verbose && ii % (N / 100) == 0
            println(string(round(ii / N * 1e2), "% completed; cost = ", cost(k, order, p=p, w_even=w_even)))
            flush(stdout)
        end
    end
    return order
end