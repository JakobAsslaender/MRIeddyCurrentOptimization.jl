
function cost(kv, x, w_exp, w_even)
    xv = vec(x);
    dk = kv[:,xv[1:end - 1]] - kv[:,x[2:end]];
    dk = reduce(+, dk.^2, dims=1).^w_exp;
    c = sum(dk[1:2:end]) + w_even * sum(dk[2:2:end]);    
    return c
end


function cost_part(kv, x, iFA, iC1, iC2, w_exp, w_even)
    if iFA % 2 == 0
        w12 = 1;
        w23 = w_even;
    else
        w12 = w_even;
        w23 = 1;
    end

    i1 = (iC1 - 1) * nFA + iFA;
    i2 = (iC2 - 1) * nFA + iFA;

    xv = vec(x);
    x1 = xv[i1 - 1:i1 + 1];
    x2 = xv[i2 - 1:i2 + 1];
    co  = w12 * reduce(+, (kv[:,x1[1]] - kv[:,x1[2]]).^2, dims=1).^w_exp;
    co += w23 * reduce(+, (kv[:,x1[2]] - kv[:,x1[3]]).^2, dims=1).^w_exp;
    co += w12 * reduce(+, (kv[:,x2[1]] - kv[:,x2[2]]).^2, dims=1).^w_exp;
    co += w23 * reduce(+, (kv[:,x2[2]] - kv[:,x2[3]]).^2, dims=1).^w_exp;

    cn  = w12 * reduce(+, (kv[:,x1[1]] - kv[:,x2[2]]).^2, dims=1).^w_exp;
    cn += w23 * reduce(+, (kv[:,x2[2]] - kv[:,x1[3]]).^2, dims=1).^w_exp;
    cn += w12 * reduce(+, (kv[:,x2[1]] - kv[:,x1[2]]).^2, dims=1).^w_exp;
    cn += w23 * reduce(+, (kv[:,x1[2]] - kv[:,x2[3]]).^2, dims=1).^w_exp;

    c = cn - co;
    return c[1]
end

function cost_part_pairs(kv, x, iFA, iC1, iC2, w_exp)

    i1 = (iC1 - 1) * nFA + iFA;
    i2 = (iC2 - 1) * nFA + iFA;

    xv = vec(x);
    co  = reduce(+, (kv[:,xv[i1 - 1]] - kv[:,xv[i1]]).^2, dims=1).^w_exp;
    co += reduce(+, (kv[:,xv[i1 + 1]] - kv[:,xv[i1 + 2]]).^2, dims=1).^w_exp;
    co += reduce(+, (kv[:,xv[i2 - 1]] - kv[:,xv[i2]]).^2, dims=1).^w_exp;
    co += reduce(+, (kv[:,xv[i2 + 1]] - kv[:,xv[i2 + 2]]).^2, dims=1).^w_exp;

    cn  = reduce(+, (kv[:,xv[i1 - 1]] - kv[:,xv[i2]]).^2, dims=1).^w_exp;
    cn += reduce(+, (kv[:,xv[i2 + 1]] - kv[:,xv[i1 + 2]]).^2, dims=1).^w_exp;
    cn += reduce(+, (kv[:,xv[i2 - 1]] - kv[:,xv[i1]]).^2, dims=1).^w_exp;
    cn += reduce(+, (kv[:,xv[i1 + 1]] - kv[:,xv[i2 + 2]]).^2, dims=1).^w_exp;

    c = cn - co;
    return c[1]
end


function SimulatedAnneling_fast(kv, x, N, nFA, nCyc, w_exp, w_even)
    rng = MersenneTwister(12345);
    for ii = 1:N
        iFA = Int32.(ceil.(rand(rng) * nFA));
        iC1 = Int32.(ceil.(rand(rng) * nCyc));
        iC2 = Int32.(ceil.(rand(rng) * nCyc));

        if iFA == 1 && (iC1 == 1 || iC2 == 1)
            iFA = 2;
        elseif iFA == nFA && (iC1 == nCyc || iC2 == nCyc)
            iFA = nFA - 1;
        end

        c = cost_part(kv, x, iFA, iC1, iC2, w_exp, w_even);
        if exp(-c / (1 - ii / N)^6) > rand(rng)
            x2 = x[iFA,iC2];
            x[iFA,iC2] = x[iFA,iC1];
            x[iFA,iC1] = x2;
        end
        if ii % (N / 1000) == 0
            println(string(round(ii / N * 1e3) / 10, " completed; cost = ", cost(kv, x, w_exp, w_even)))
            flush(stdout)
        end
    end
    return x
end


function SimulatedAnneling_Pairs(kv, x, N, nFA, nCyc, w_exp)
    rng = MersenneTwister(12345);
    for ii = 1:N
        iFA = Int32.(ceil.(rand(rng) * nFA / 2) * 2);
        iC1 = Int32.(ceil.(rand(rng) * nCyc));
        iC2 = Int32.(ceil.(rand(rng) * nCyc));

        if iFA == nFA && (iC1 == nCyc || iC2 == nCyc)
            iFA = nFA - 2;
        end

        c = cost_part_pairs(kv, x, iFA, iC1, iC2, w_exp);
        if exp(-c / (1 - ii / N)^6) > rand(rng)
            if iFA == nFA
                x21 = x[iFA,iC2];
                x22 = x[1,iC2+1];
                x[iFA,iC2] = x[iFA,iC1];
                x[1,iC2+1] = x[1,iC1+1];
                x[iFA,iC1] = x21;
                x[1,iC1+1] = x22;
            else
                x2 = x[iFA:iFA + 1,iC2];
                x[iFA:iFA + 1,iC2] = x[iFA:iFA + 1,iC1];
                x[iFA:iFA + 1,iC1] = x2;
            end
        end
        if ii % (N / 1000) == 0
            println(string(round(ii / N * 1e3) / 10, " completed; cost = ", cost(kv, x, w_exp, 1)))
            flush(stdout)
        end
    end
    return x
end

# function SimulatedAnneling_WholeCost(kv, x, N, nFA, nCyc)
#     rng = MersenneTwister(12345);
#     c = cost(kv, x);
#     for ii = 1:N
#         iFA = Int32.(ceil.(rand(rng) * nFA));
#         iC1 = Int32.(ceil.(rand(rng) * nCyc));
#         iC2 = Int32.(ceil.(rand(rng) * nCyc));
        
#         if iFA == 1 && (iC1 == 1 || iC2 == 1)
#             iFA = 2;
#         elseif iFA == nFA && (iC1 == nCyc || iC2 == nCyc)
#             iFA = nFA-1;
#         end

#         xn = copy(x);
#         xn[iFA,iC1] = x[iFA,iC2];
#         xn[iFA,iC2] = x[iFA,iC1];
#         cn = cost(kv, xn);
#         if exp(-(cn - c) / (1 - ii / N)^6) > rand(rng) 
#             x = xn;
#             c = copy(cn);
#         end
#     end
#     return x
# end