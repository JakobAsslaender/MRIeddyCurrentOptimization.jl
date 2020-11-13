clear
nFA = 1141;
nCyc = 189;

M = [0 1 0; 0 0 1; 1 0 1];
[v, ~] = eig(M);
GA1 = v(1,1) / v(end,1);
GA2 = v(2,1) / v(end,1);

theta = acos(mod((0:(nCyc*nFA-1))*GA1, 1));
phi = (0:(nCyc*nFA-1))*2*pi*GA2;

theta = reshape(theta, nCyc, nFA);
phi   = reshape(phi  , nCyc, nFA);
% theta(:,end) = circshift(theta(:,1), [-1 0]);
% phi  (:,end) = circshift(phi  (:,1), [-1 0]);
theta = theta(:).';
phi   = phi  (:).';

clear k
k(3,:) = cos(theta);
k(2,:) = sin(theta) .* sin(phi);
k(1,:) = sin(theta) .* cos(phi);
k = reshape(k, 3, [], nFA);
k = permute(k, [1 3 2]);

%%
ks = k(:,:);
dk = l2_norm(ks(:,1:end-1) - ks(:,2:end), 1)/pi*180;
figure(1); clf; 
subplot(3,1,1); hold on
histogram(dk(1:2:end), 1:120)
histogram(dk(2:2:end), 1:120)
xlim([0 120])
ylim([0 4e4])
title('Original golden means pattern (reordered to first sampled the 189 cycles)')

load ~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_GA_Result_189_1141_6_10_1E+10.mat
ks = k(:,:);
ks = ks(:,xopt(:));
ks = reshape(ks, 3, nFA, nCyc);
ks(:,end+1,:) = circshift(ks(:,1,:), [0 0 -1]);
ks = ks(:,:);

dk = l2_norm(ks(:,1:end-1) - ks(:,2:end), 1)/pi*180;
subplot(3,1,2); hold on
histogram(dk(1:2:end), 1:120)
histogram(dk(2:2:end), 1:120)
xlim([0 120])
ylim([0 4e4])
title('optimized with even weighting = 10 x odd weighting')

% load ~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_GA_Result_189_1141_6_100_1E+10.mat
load ~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_GA_Result_189_1141_6_1_1e+10.mat
ks = k(:,:);
ks = ks(:,xopt(:));
ks = reshape(ks, 3, nFA, nCyc);
ks(:,end+1,:) = circshift(ks(:,1,:), [0 0 -1]);
ks = ks(:,:);

dk = l2_norm(ks(:,1:end-1) - ks(:,2:end), 1)/pi*180;
subplot(3,1,3); hold on
histogram(dk(1:2:end), 1:120)
histogram(dk(2:2:end), 1:120)
xlim([0 120])
ylim([0 4e4])
title('optimized with even weighting = 1 x odd weighting')
xlabel('k-space angle increment (deg)');
legend('odd jumps', 'even jumps')