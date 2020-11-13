##
using BenchmarkTools
using LinearAlgebra
using Random
using MAT
using Printf
include("SimmulatedAnnealingFunctions.jl")

##
file = matopen(expanduser(string("~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/Phi_Theta_array_No_Pair_Full_Sphere_nFA_1142_nCyc_189.mat")), "r")
theta = read(file, "theta")
phi = read(file, "phi")
close(file)

nFA = size(phi,2);
nCyc = size(phi,1);

theta = vec(permutedims(theta, (2,1)));
phi = vec(permutedims(phi, (2,1)));


k = zeros(3, length(theta));
k[3,:] = cos.(theta);
k[2,:] = sin.(theta) .* sin.(phi);
k[1,:] = sin.(theta) .* cos.(phi);

##
w_exp = 6;
# w_even = 100;
w_even = parse(Int32, ENV["SLURM_ARRAY_TASK_ID"])
# N = Int64.(1e9);
N = Int64.(10^parse(Int64, ENV["Iter_exp"]));
println(string("N = ", N))
println(string("w_even = ", w_even))

xinit = Int32.(1:(nCyc*nFA));
xinit = reshape(xinit, nFA, nCyc);
println(cost(k,xinit,w_exp/2,w_even));
# xopt = SimulatedAnneling(kv, xinit, N, nFA, nCyc)
xopt = SimulatedAnneling_fast(k, xinit, N, nFA, nCyc,w_exp/2,w_even);
cost(k,xopt,w_exp/2,w_even)


## Save Data
file = matopen(expanduser(string("~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_Poisson_Full_Disc_Result_", nCyc, "_", nFA, "_", w_exp, "_", w_even, "_", @sprintf("%.0E", N), ".mat")), "w")
write(file, "xopt", xopt)
close(file)

