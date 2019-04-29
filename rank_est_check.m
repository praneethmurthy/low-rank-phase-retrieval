%%  Attempts to speed up LRPR new algorithm

%close all
clear;
clc;


tt1 = tic;
Params.Tmont = 10;

Params.n  =  200;   % Number of rows of the low rank matrix
Params.q  =  150;   % Number of columns of the matrix for LRPR
Params.r  =  4;     % Rank
Params.m = 80;     % Number of measurements

Params.tnew = 5;    % Total number of main loops of new LRPR
Params.told = 5;    % Total number of main loops of Old LRPR

Params.m_b = Params.m;          %Number of measuremnets for coefficient estimate
Params.m_u = Params.m;           % Number of measuremnets for subspace estimate
Params.m_init = Params.m;       % Number of measuremnets for init of subspace
%m_init = 50;

%Params.m  =  m_init + (m_b+m_u)*Params.tot;% Number of measurements

%%~PN editing m, n, r so that the variables are globally same
% TWF Parameters
Paramsrwf.m  =  Params.m;% Number of measurements
Paramsrwf.n  =  Params.n;% size of columns of coefficient matrix or x_k
Paramsrwf.r  =  Params.r;% size of columns of coefficient matrix or b_k
Paramsrwf.npower_iter = 100;% Number of loops for initialization of TWF with power method
Paramsrwf.mu          = 0.2;% Parameter for gradient
Params.Tb_LRPRnew    = unique(ceil(linspace(5, 12, Params.tnew)));% Number of loops for b_k with simple PR
%Params.Tb_LRPRnew    = 85 * ones(1, Params.tnew);
% Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
Paramsrwf.TRWF           = 85;% Number of loops for b_k with simple PR
Paramsrwf.cplx_flag   = 0;
% Paramstwf.alpha_y     = 3;
% Paramstwf.alpha_h     = 5;
% Paramstwf.alpha_ub    = 5;
% Paramstwf.alpha_lb    = 0.3;
% Paramstwf.grad_type   = 'TWF_Poiss';
%Params.seed = rng;
err_SE_iter = zeros(3, Params.tnew, Params.Tmont);

file_name = strcat(['Copmare_n', num2str(Params.n), 'm', num2str(Params.m), 'r', num2str(Params.r), 'q', num2str(Params.q)]);
file_name_txt = strcat(file_name,'.txt');
file_name_mat = strcat(file_name,'.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Generating U and B and X
%rng('shuffle')
U       =   orth(randn(Params.n, Params.r));
B       =   randn(Params.r, Params.q);
X       =   U * B;
normX  =  norm(X,'fro')^2; % Computing Frobenius norm of the low rank matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Compare

TmpErXRWF      =   zeros(Paramsrwf.TRWF,Params.Tmont);
TmpErURWF      =   zeros(Paramsrwf.TRWF,Params.Tmont);
TmpExTRWF      =   zeros(Paramsrwf.TRWF,Params.Tmont);

TmpErXLRPROld  =   zeros(Params.told,Params.Tmont);
TmpErULRPRoLd  =   zeros(Params.told,Params.Tmont);


TmpErXLRPRnew  =   zeros(Params.tnew,Params.Tmont);
TmpErULRPRnew  =   zeros(Params.tnew,Params.Tmont);
TmpExTLRPEnew  =   zeros(Params.tnew,Params.Tmont);

TmpErXLRPRqr   =   zeros(Params.tnew,Params.Tmont);
TmpErULRPRqr   =   zeros(Params.tnew,Params.Tmont);
TmpExTLRPRqr   =   zeros(Params.tnew,Params.Tmont);

sig_star = svds(X, Params.r);

est_gap1 = zeros(1, Params.Tmont);
est_gap2 = zeros(1, Params.Tmont);

for t = 1 : Params.Tmont
    fprintf('=============== Monte Carlo = %d ====================\n', t);
    [Ysqrt,Y,A] = Generate_Mes(X,Params,Params.m_init);
    
    %%checking rank estimation
    Yu      =   zeros(Params.n, Params.n);
    normest = 9/(Params.m_init * Params.q) * sum(Y(:));
    %normest = 0;
        for nh = 1 : Params.q
            %normest = sqrt((13/Params.m_init) * Y(:,nh)' * Y(:, nh));
            Ytr = Y(:,nh) .* (abs(Y(:, nh)) > normest);
            Yu  =   Yu + A(:,:,nh) * diag(Ytr) * A(:,:,nh)';
        end
        Yu      =   Yu / Params.q / Params.m_init;
        [P,sig_init,~] =   svd(Yu);
        sig_init = diag(sig_init);
        eig_gaps = sig_init(1:end-1) - sig_init(2:end);
        true_rank = Params.r;
        [~, arg_max_est] = max(eig_gaps);
        est_gap1(t) = arg_max_est;
         figure
%         subplot(211)
         plot(sig_init)
        %sig_init(true_rank) - sig_init(rank+1)
        %.5 * -1 * log(Params.m/(Params.n * Params.q))* min(sig_star)^2/Params.q
%         tmp1 = 1.* (sig_init(1:end-1) - sig_init(end) >= .5 * -1 * log(Params.m/(Params.n * Params.q))* min(sig_star)^2/Params.q);
sig_init(end)
min(sig_init)
tmp1 = 1.* (sig_init(1:end-1) - sig_init(end) >= 2* min(sig_star)^2/Params.q);
        if (all(tmp1 == 0))
            max_min_gap_est = 1;
        else
            max_min_gap_est = find(tmp1 == 1, 1, 'last');
        end
        
        est_gap2(t) = max_min_gap_est;
%        subplot(212)
%        plot(tmp1)
end

figure
subplot(211)
hist(est_gap1)
title('gap based')
subplot(212)
hist(est_gap2)
title('no trunc')
