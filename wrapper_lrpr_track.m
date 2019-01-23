%%  Attempts to speed up LRPR new algorithm

%close all
clear;
clc;


tt1 = tic;
Params.Tmont = 3;

Params.n  =  300;   % Number of rows of the low rank matrix
Params.q  =  5100;   % Number of columns of the matrix for LRPR
Params.r  =  2;     % Rank
Params.m = 100;     % Numlber of measurements
Params.alpha = 250;
Params.L = 8;
Params.thresh = .6;

Params.tnew = 10;    % Total number of main loops of new LRPR
Params.told = 10;    % Total number of main loops of Old LRPR

Params.m_b = Params.m;          %Number of measuremnets for coefficient estimate
Params.m_u = Params.m;           % Number of measuremnets for subspace estimate
Params.m_init = Params.n;       % Number of measuremnets for init of subspace
%m_init = 50;

%Params.m  =  m_init + (m_b+m_u)*Params.tot;% Number of measurements

%%~PN editing m, n, r so that the variables are globally same
% TWF Parameters
Paramsrwf.m  =  Params.m;% Number of measurements
Paramsrwf.n  =  Params.n;% size of columns of coefficient matrix or x_k
Paramsrwf.r  =  Params.r;% size of columns of coefficient matrix or b_k
Paramsrwf.npower_iter = 50;% Number of loops for initialization of TWF with power method
Paramsrwf.mu          = 0.2;% Parameter for gradient
%Params.Tb_LRPRnew    = unique(ceil(linspace(30, 100, Params.tnew)));% Number of loops for b_k with simple PR
Params.Tb_LRPRnew    = 85 * ones(1, Params.tnew);
% Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
Paramsrwf.TRWF           = 25;% Number of loops for b_k with simple PR
Paramsrwf.cplx_flag   = 0;
% Paramstwf.alpha_y     = 3;
% Paramstwf.alpha_h     = 5;
% Paramstwf.alpha_ub    = 5;
% Paramstwf.alpha_lb    = 0.3;
% Paramstwf.grad_type   = 'TWF_Poiss';
%Params.seed = rng;
% err_SE_iter = zeros(3, Params.tnew, Params.Tmont);

%err_SE_iter = zeros(Params.tnew, Params.Tmont);

% file_name = strcat(['Copmare_n', num2str(Params.n), 'm', num2str(Params.m), 'r', num2str(Params.r), 'q', num2str(Params.q)]);
% file_name_txt = strcat(file_name,'.txt');
% file_name_mat = strcat(file_name,'.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Generating U and B and X
t_1 = 2992;
U0       =   orth(randn(Params.n, Params.r));
Mse = randn(Params.n);
Mse1 = (Mse - Mse')/2;
U1 = expm(0.001 * Mse1) * U0;

B       =   randn(Params.r, Params.q);
X       =   [U0 * B(:, 1 : t_1), U1* B(:, t_1 + 1 : end)];
Params.sig_star = svds(X, rank(X));
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


for t = 1 : Params.Tmont
    
    fprintf('=============== Monte Carlo = %d ====================\n', t);
    [Ysqrt,Y,A] = Generate_Mes(X,Params,Params.m);
    tic;
    [B_new_sample, U_new_sample, U_track_new, t_calc] = ...
        LRPR_track(Params, Paramsrwf, Y, Ysqrt, A, X);
    TmpTLRPmes(t) = toc;
    
    tic;
    [B_new_sample_large, U_new_sample_large, U_track_new_large, t_calc_large] = ...
        LRPR_track_large(Params, Paramsrwf, Y, Ysqrt, A, X);
    TmpTLRPmes_large(t) = toc;
    ERULRPRmes_large(t) = eps;
    ERULRPRmes(t) = eps;
    %ERULRPRmes(t)  =  abs(sin(subspace(U_new_sample, U)));
    %fprintf('LRPR tracking error U:\t %2.2e\t\t Time:\t %2.2e\n', ERULRPRmes(t), TmpTLRPmes(t));
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % Error X
    %%%%%%%%%%%%
    Error_X_LRPR_Newmes = 0;
    
    for ii = 1 : length(U_track_new)
        if(t_calc(ii) <= t_1)
            err_SE_iter(ii, t) = abs(sin(subspace(U_track_new{ii}, U0)));
        else
            err_SE_iter(ii, t) = abs(sin(subspace(U_track_new{ii}, U1)));
        end
    end
    
    for ii = 1 : length(U_track_new_large)
        if(t_calc_large(ii) <= t_1)
            err_SE_iter_large(ii, t) = abs(sin(subspace(U_track_new_large{ii}, U0)));
        else
            err_SE_iter_large(ii, t) = abs(sin(subspace(U_track_new_large{ii}, U1)));
        end
    end
end

mean_Error_U_LRPR_Newmes = mean(ERULRPRmes);

mean_Time_LRPR_Newmes = mean(TmpTLRPmes);

mean_Error_U_LRPR_Newmes_large = mean(ERULRPRmes_large);

mean_Time_LRPR_Newmes_large = mean(TmpTLRPmes_large);

fprintf('**************************************\n');
fprintf('Error U: ... \n');
fprintf('LRPR theory:\t%2.2e\n', mean_Error_U_LRPR_Newmes);
fprintf('**************************************\n');
fprintf('Exe Time: ... \n');
fprintf('LRPR theory:\t%2.2e\n', mean_Time_LRPR_Newmes);
toc(tt1)


final_err_SE = mean(err_SE_iter, 2);
final_err_SE_large = mean(err_SE_iter_large, 2);
% final_err_SE_med = median(err_SE_iter, 3);
% final_err_SE_std = std(err_SE_iter, 0, 3);

figure;
plot(t_calc, log10(final_err_SE), 'rs--', 'LineWidth', 2);
hold
plot(t_calc_large, log10(final_err_SE_large), 'b>:', 'LineWidth', 2);
axis tight
l1 = legend('PST-all', 'PST-large');
set(l1, 'Fontsize', 15);
stry = '$$\log(SE(\hat{U}^t, U))$$';
xlabel('time (t)', 'Fontsize', 15)
ylabel(stry, 'Interpreter', 'latex', 'Fontsize', 15)
title('SE = 0.8', 'Fontsize', 15)
%save('track_se08_mc30_smallm.mat')
