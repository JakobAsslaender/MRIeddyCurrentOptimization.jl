##
using BenchmarkTools
using LinearAlgebra
using Random
using MAT
using Printf
include("SimmulatedAnnealingFunctions.jl")

##
# nFA = 1141;
# nCyc = 189;
nFA = 571;
nCyc = 872;

M = [0 1 0; 0 0 1; 1 0 1];
s, v = eigen(M);
GA1 = real(v[1,end] / v[end,end]);
GA2 = real(v[2,end] / v[end,end]);

theta = acos.(((0:(nCyc * nFA - 1)) * GA1) .% 1);
phi = Float64.(0:(nCyc * nFA - 1)) * 2 * pi * GA2;

theta = reshape(theta, nCyc, nFA);
phi   = reshape(phi, nCyc, nFA);
# theta[:,end] = circshift(theta[:,1], -1);
# phi[:,end] = circshift(phi[:,1], -1);
theta = vec(theta);
phi   = vec(phi);

k = zeros(3, length(theta));
k[3,:] = cos.(theta);
k[2,:] = sin.(theta) .* sin.(phi);
k[1,:] = sin.(theta) .* cos.(phi);
# k = k / pi; # dk in the range of 0:1

k = reshape(k, 3, nCyc, nFA);
k = permutedims(k, (1, 3, 2));
kv = reshape(k, 3, nCyc*nFA);


##
# N = Int64.(1e10);
N = parse(Int32, ENV["SLURM_ARRAY_TASK_ID"])
N = Int64.(10^N);
w_exp = 6;
w_even = 1;
# w_even = 100;

xinit = Int32.(1:(nCyc*nFA))
xinit = reshape(xinit, nFA, nCyc)
println(cost(kv,xinit,w_exp/2,w_even))
# xopt = SimulatedAnneling(kv, xinit, N, nFA, nCyc)
xopt = SimulatedAnneling_fast!(kv, xinit, N, nFA, nCyc,w_exp/2,w_even)
cost(kv,xopt,w_exp/2,w_even)


## Save Data
file = matopen(expanduser(string("~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_GA_Result_", nCyc, "_", nFA, "_", w_exp, "_", w_even, "_", @sprintf("%.0E", N), ".mat")), "w")
write(file, "xopt", xopt)
close(file)

