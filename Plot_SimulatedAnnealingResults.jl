##
using Plots

file = matopen(expanduser(string("~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/Phi_Theta_array_No_Pair_Full_Sphere_nFA_1142_nCyc_189.mat")), "r")
theta = read(file, "theta")
phi = read(file, "phi")
close(file)

nFA = size(phi,2);
nCyc = size(phi,1);
w_exp = 6;
w_even = 100;
N = 1e9;

file = matopen(expanduser(string("~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_Poisson_Full_Disc_Result_", nCyc, "_", nFA, "_", w_exp, "_", w_even, "_", @sprintf("%.0E", N), ".mat")), "r")
xopt = read(file, "xopt", xopt)
close(file)

##
using Plots
for i in 1:20
    # IJulia.clear_output(true)
    x = 0:.01:4; y = sin.(i*x)
    Plots.display(plot(x,y, color="red"))
    read(stdin, 1);
end
println("Done!")