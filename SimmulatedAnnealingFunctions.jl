
function cost(kv, x, w_exp, w_even)
    xv = vec(x);
    dk = kv[:,xv[1:end - 1]] - kv[:,x[2:end]];
    dk = reduce(+, dk.^2, dims=1).^w_exp;
    c = sum(dk[1:2:end]) + w_even * sum(dk[2:2:end]);    
    return c
end


function cost_part(kv, x, iFA, nFA, iC1, iC2, w_exp, w_even)
    if iFA % 2 == 0
        w12 = 1
        w23 = w_even
    else
        w12 = w_even
        w23 = 1
    end

    i1 = (iC1 - 1) * nFA + iFA
    i2 = (iC2 - 1) * nFA + iFA

    x11 = x[i1 - 1]
    x12 = x[i1]
    x13 = x[i1 + 1]
    x21 = x[i2 - 1]
    x22 = x[i2]
    x23 = x[i2 + 1]

    Δk  = (kv[1,x11] - kv[1,x12])^2
    Δk += (kv[2,x11] - kv[2,x12])^2
    Δk += (kv[3,x11] - kv[3,x12])^2
    co = w12 * Δk^w_exp

    Δk  = (kv[1,x11] - kv[1,x22])^2
    Δk += (kv[2,x11] - kv[2,x22])^2
    Δk += (kv[3,x11] - kv[3,x22])^2
    cn = w12 * Δk^w_exp


    Δk  = (kv[1,x12] - kv[1,x13])^2
    Δk += (kv[2,x12] - kv[2,x13])^2
    Δk += (kv[3,x12] - kv[3,x13])^2
    co += w23 * Δk^w_exp

    Δk  = (kv[1,x22] - kv[1,x13])^2
    Δk += (kv[2,x22] - kv[2,x13])^2
    Δk += (kv[3,x22] - kv[3,x13])^2
    cn += w23 * Δk^w_exp


    Δk  = (kv[1,x21] - kv[1,x22])^2
    Δk += (kv[2,x21] - kv[2,x22])^2
    Δk += (kv[3,x21] - kv[3,x22])^2
    co += w12 * Δk^w_exp

    Δk  = (kv[1,x21] - kv[1,x12])^2
    Δk += (kv[2,x21] - kv[2,x12])^2
    Δk += (kv[3,x21] - kv[3,x12])^2
    cn += w12 * Δk^w_exp


    Δk  = (kv[1,x22] - kv[1,x23])^2
    Δk += (kv[2,x22] - kv[2,x23])^2
    Δk += (kv[3,x22] - kv[3,x23])^2
    co += w12 * Δk^w_exp

    Δk  = (kv[1,x12] - kv[1,x23])^2
    Δk += (kv[2,x12] - kv[2,x23])^2
    Δk += (kv[3,x12] - kv[3,x23])^2
    cn += w12 * Δk^w_exp

    return cn - co
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


function SimulatedAnneling_fast!(k_vec, order, N, nFA, nCyc, w_exp, w_even; rng = MersenneTwister(12345))
    for ii = 1:N
        iFA = Int32.(ceil.(rand(rng) * nFA))
        iC1 = Int32.(ceil.(rand(rng) * nCyc))
        iC2 = Int32.(ceil.(rand(rng) * nCyc))

        if iFA == 1 && (iC1 == 1 || iC2 == 1)
            iFA = 2
        elseif iFA == nFA && (iC1 == nCyc || iC2 == nCyc)
            iFA = nFA - 1
        end

        c = cost_part(k_vec, order, iFA, nFA, iC1, iC2, w_exp, w_even)
        if exp(-c / (1 - ii / N)^6) > rand(rng)
            x2 = order[iFA,iC2]
            order[iFA,iC2] = order[iFA,iC1]
            order[iFA,iC1] = x2
        end
        # if ii % (N / 100) == 0
        #     println(string(round(ii / N * 1e2), "% completed; cost = ", cost(k_vec, order, w_exp, w_even)))
        #     flush(stdout)
        # end
    end
    return order
end


function SimulatedAnneling_Pairs(kv, order, N, nFA, nCyc, w_exp)
    rng = MersenneTwister(12345);
    for ii = 1:N
        iFA = Int32.(ceil.(rand(rng) * nFA / 2) * 2);
        iC1 = Int32.(ceil.(rand(rng) * nCyc));
        iC2 = Int32.(ceil.(rand(rng) * nCyc));

        if iFA == nFA && (iC1 == nCyc || iC2 == nCyc)
            iFA = nFA - 2;
        end

        c = cost_part_pairs(kv, order, iFA, iC1, iC2, w_exp);
        if exp(-c / (1 - ii / N)^6) > rand(rng)
            if iFA == nFA
                x21 = order[iFA,iC2];
                x22 = order[1,iC2+1];
                order[iFA,iC2] = order[iFA,iC1];
                order[1,iC2+1] = order[1,iC1+1];
                order[iFA,iC1] = x21;
                order[1,iC1+1] = x22;
            else
                x2 = order[iFA:iFA + 1,iC2];
                order[iFA:iFA + 1,iC2] = order[iFA:iFA + 1,iC1];
                order[iFA:iFA + 1,iC1] = x2;
            end
        end
        if ii % (N / 1000) == 0
            println(string(round(ii / N * 1e3) / 10, " completed; cost = ", cost(kv, order, w_exp, 1)))
            flush(stdout)
        end
    end
    return order
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