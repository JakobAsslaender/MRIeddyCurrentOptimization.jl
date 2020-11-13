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
load ~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_GA_all_Result_189_1141_1e10.mat
% load ~/mygs/asslaj01/20201019_SimmulatedAnnealing_Traj_EddyCurrents/SimAnnealing_all_Result_189_1141_3_5_1E+08.mat

%%
ks = k(:,:);
dk = l2_norm(ks(:,1:end-1) - ks(:,2:end), 1)/pi*180;
figure(1); clf; hold on
plot(dk(1:2:end), 'xb')
plot(dk(2:2:end), 'xr')

ks = ks(:,xopt(:));
ks = reshape(ks, 3, nFA, nCyc);
ks(:,end+1,:) = circshift(ks(:,1,:), [0 0 -1]);
ks = ks(:,:);

dk = l2_norm(ks(:,1:end-1) - ks(:,2:end), 1)/pi*180;
figure(3); clf; hold on
plot(dk(1:2:end), 'xb')
plot(dk(2:2:end), 'xr')
