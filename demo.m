clear all; clc;
addpath(genpath(pwd));
randn('state',0);  rand('state',0);

%% Generate noise-free nonnegative tensor
M = 20; N = 50; P = 100; DIM = [M,N,P]; R = 10;
Z_true = cell(length(DIM),1);
for m=1:length(DIM)
    Z_true{m} = rand(DIM(m),R);
end
X_true = double(ktensor(Z_true));

%% Add noise
SNR = 15;
sigma2 = norm(X_true(:))^2*(1/(M*N*P))*(1/(10^(SNR/10)));
GN = sqrt(sigma2)*randn(DIM);
Y = X_true + GN;

%% Algorithm
tic;
model = fast_nn_ptcpd(Y,min(DIM));
running_time = toc;
