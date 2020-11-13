%%
clear
load ~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/Phi_Theta_array_pair_3_nFA_1142_nCyc_189.mat
nFA  = size(phi,2);
nCyc = size(phi,1);

theta = col(theta.');
phi   = col(phi.');

clear k
k(3,:) = cos(theta);
k(2,:) = sin(theta) .* sin(phi);
k(1,:) = sin(theta) .* cos(phi);

ks = k(:,:);
dk = l2_norm(ks(:,1:end-1) - ks(:,2:end), 1)/pi*180;
figure(1); clf; 
subplot(3,1,1); hold on
histogram(dk(1:2:end), 1:120)
histogram(dk(2:2:end), 1:120)
xlim([0 120])
ylim([0 1e4])
title('Original paired poisson disc pattern')

load ~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_Poisson_Result_Pairs_189_1142_6_1E+08.mat
ks = k(:,:);
ks = ks(:,xopt(:));
dk = l2_norm(ks(:,1:end-1) - ks(:,2:end), 1)/pi*180;
subplot(3,1,2); hold on
histogram(dk(1:2:end), 1:120)
histogram(dk(2:2:end), 1:120)
xlim([0 120])
ylim([0 1e4])
title('optimized for 1e8 iterations')

load ~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_Poisson_Result_Pairs_189_1142_6_1E+09.mat
ks = k(:,:);
ks = ks(:,xopt(:));
dk = l2_norm(ks(:,1:end-1) - ks(:,2:end), 1)/pi*180;
subplot(3,1,3); hold on
histogram(dk(1:2:end), 1:120)
histogram(dk(2:2:end), 1:120)
xlim([0 120])
ylim([0 1e4])
title('optimized for 1e9 iterations')
xlabel('k-space angle increment (deg)');
legend('odd jumps', 'even jumps')
