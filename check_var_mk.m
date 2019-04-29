%%  Attempts to speed up LRPR new algorithm

%close all
clear;
clc;


tt1 = tic;
Params.Tmont = 1;

Params.n  =  200;   % Number of rows of the low rank matrix
Params.q  =  400;   % Number of columns of the matrix for LRPR
Params.r  =  4;     % Rank
% Params.m = 80;     % Number of measurements

% Params.m = 80 * ones(Params.q, 1);     % Number of measurements
%Params.m = ceil(linspace(80, 30, Params.q));     % Number of measurements

Params.m = ceil(10 * randn(Params.q, 1)) + 80;
Params.tnew = 10;    % Total number of main loops of new LRPR
Params.told = 10;    % Total number of main loops of Old LRPR

Params.m_b = Params.m(1);          %Number of measuremnets for coefficient estimate
Params.m_u = Params.m(1);           % Number of measuremnets for subspace estimate
Params.m_init = Params.m(1);       % Number of measuremnets for init of subspace
%m_init = 50;
Params.rank_est_flag = 0;

%Params.m  =  m_init + (m_b+m_u)*Params.tot;% Number of measurements

%%~PN editing m, n, r so that the variables are globally same
% TWF Parameters
Paramsrwf.m  =  Params.m(1);% Number of measurements
Paramsrwf.n  =  Params.n;% size of columns of coefficient matrix or x_k
Paramsrwf.r  =  Params.r;% size of columns of coefficient matrix or b_k
Paramsrwf.npower_iter = 100;% Number of loops for initialization of TWF with power method
Paramsrwf.mu          = 0.2;% Parameter for gradient
Params.Tb_LRPRnew    = unique(ceil(linspace(5, 20, Params.tnew)));% Number of loops for b_k with simple PR
%Params.Tb_LRPRnew    = 85 * ones(1, Params.tnew);
% Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
Paramsrwf.TRWF           = 300;% Number of loops for b_k with simple PR
err_rwf = zeros(Paramsrwf.TRWF + 1, Params.q);
Paramsrwf.cplx_flag   = 0;
% Paramstwf.alpha_y     = 3;
% Paramstwf.alpha_h     = 5;
% Paramstwf.alpha_ub    = 5;
% Paramstwf.alpha_lb    = 0.3;
% Paramstwf.grad_type   = 'TWF_Poiss';
%Params.seed = rng;
err_SE_iter = zeros(2, Params.tnew, Params.Tmont);
err_X_iter = zeros(2, Params.tnew, Params.Tmont);
err_X_rwf_iter = zeros(Paramsrwf.TRWF+1, Params.Tmont);

%file_name = strcat(['Copmare_n', num2str(Params.n), 'm', num2str(Params.m), 'r', num2str(Params.r), 'q', num2str(Params.q)]);
%file_name_txt = strcat(file_name,'.txt');
%file_name_mat = strcat(file_name,'.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Generating U and B and X
%rng('shuffle')
U       =   orth(randn(Params.n, Params.r));
B       =   randn(Params.r, Params.q);
X       =   U * B;
normX  =  norm(X,'fro')^2; % Computing Frobenius norm of the low rank matrix

Params.sig_star = svds(X, Params.r);
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

time_QR = zeros(Params.tnew, Params.Tmont);
time_OLD = zeros(Params.told, Params.Tmont);
time_RWF = zeros(Paramsrwf.TRWF, Params.Tmont);

for t = 1 : Params.Tmont
    Ysqrt = cell(Params.q, 1);
    Y = cell(Params.q, 1);
    A = cell(Params.q, 1);
    %Params.m = ceil(10 * randn(Params.q, 1)) + 80;
    %[Ysqrt,Y,A] = Generate_Mes(X,Params,Params.m);
    
    for qq = 1 : Params.q
        A{qq} = randn(Params.n, Params.m(qq));
        Y{qq} = abs(A{qq}' * X(:, qq)).^2;
        Ysqrt{qq} = sqrt(Y{qq});
    end
    fprintf('=============== Monte Carlo = %d ====================\n', t);
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % LRPR - practice
    %%%%%%%%%%%%%
    tic;
    [B_QR, U_QR, X_hat_QR, U_track_QR, X_track_QR, time_QR(:, t)] = LRPRQR1(Params, Paramsrwf, Y, Ysqrt, A);
    TmpTLRPQR(t) = toc;
    ERULRPRQR(t)  =  abs(sin(subspace(U_QR, U)));
    fprintf('LRPR-practice error U:\t %2.2e\t\t Time:\t %2.2e\n', ERULRPRQR(t), TmpTLRPQR(t));
    
    TmpTLRPmes(t) = 0;
    ERULRPRmes(t)  =  eps;
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    %   LRPR Alt Min
    %%%%%%%%%%%%%
    
    tic;
    [X_old, U_old, U_track_old, X_track_OLD, time_OLD(:, t)]= LRPR_AltMin1(Y, Ysqrt, A, Params);
    TmpExTLRPROld(t) = toc;
    TmpErULRPROld(t) =  abs(sin(subspace(U_old, U)));
    fprintf('LRPR error U:\t %2.2e\t\t Time:\t %2.2e\n', TmpErULRPROld(t), TmpExTLRPROld(t));
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % RWF
    %%%%%%%%%%%%
    %     X_rwf  =  zeros(Params.n, Params.q);
    %     tic;
    %     for nk = 1: Params.q
    %         Amatrix =  A(:,:,nk)';
    %         A1    = @(I) Amatrix  * I;
    %         At    = @(Y) Amatrix' * Y;
    %         [x_rwf, err_rwf(:, nk), time_RWF(:, t)]  = RWFsimple2(Ysqrt(:,nk), Paramsrwf, A1, At, X(:, nk));
    %         X_rwf(:,nk)  =  x_rwf;
    %     end
    %     err_X_rwf_iter(:, t) = sum(err_rwf, 2)/normX;
    %     [Ur,~,~] =  svd(X_rwf);
    %     U_rwf  =  Ur(:,1:Params.r);
    %     TmpExTrwf(t)    =  toc;
    %     TmpErUrwf(t)    =  abs(sin(subspace(U_rwf, U)));
    %     fprintf('RWF subspace error:\t%2.2e\t\tTime: %2.2e\n', TmpErUrwf(t), TmpExTrwf(t));
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % Error X
    %%%%%%%%%%%%
    Error_X_LRPR_new = 0;
    Error_X_LRPR_QR = 0;
    Error_X_LRPR_Newmes = 0;
    Error_X_LRPROLD = 0;
    %     Error_X_RWF = 0;
    
    for nk = 1 : Params.q
        x_opt       =   X(:, nk);
        
        %   LRPR practical
        x_hat = X_hat_QR(:, nk);
        Error_X_LRPR_QR = Error_X_LRPR_QR + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
        
        %   LRPR theory
        %         x_hat       =   X_new_sample(:, nk);
        %         Error_X_LRPR_Newmes = Error_X_LRPR_Newmes + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
        
        %   LRPR old
        x_hat       =   X_old(:, nk);
        Error_X_LRPROLD = Error_X_LRPROLD + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
        
        %   RWF
        %         x_hat       =   U_rwf*U_rwf'*X_rwf(:, nk);
        %         Error_X_RWF = Error_X_RWF + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
    end
    tmpEr_X_LRPR_QR(t) = Error_X_LRPR_QR / normX;
    %     tmpEr_X_LRPR_Newmes(t) = Error_X_LRPR_Newmes / normX;
    tmpEr_X_LRPROLD(t) = Error_X_LRPROLD / normX;
    %     tmpEr_X_RWF(t) = Error_X_RWF / normX;
    %     for ii = 1 : Params.tnew
    %         err_SE_iter(:, ii, t) = [abs(sin(subspace(U_track_QR{ii}, U))); ...
    %             abs(sin(subspace(U_track_new{ii}, U))); ...
    %             abs(sin(subspace(U_track_old{ii}, U)));];
    %     end
    
    err_track_X_QR = zeros(Params.tnew,1);
    err_track_X_OLD = zeros(Params.tnew,1);
    for ii = 1 : Params.tnew
        err_SE_iter(:, ii, t) = [abs(sin(subspace(U_track_QR{ii}, U))); ...
            abs(sin(subspace(U_track_old{ii}, U)));];
        for jj = 1 : Params.q
            x_opt       =   X(:, nk);
            
            XhatQR = X_track_QR{ii};
            x_hat = XhatQR(:, nk);
            err_track_X_QR(ii) = err_track_X_QR(ii) + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
            
            XhatOLD = X_track_OLD{ii};
            x_hat       =   XhatOLD(:, nk);
            err_track_X_OLD(ii) = err_track_X_OLD(ii) + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
        end
        err_track_X_OLD(ii) = err_track_X_OLD(ii)/Params.q;
        err_track_X_QR(ii) = err_track_X_QR(ii)/Params.q;
        
        err_X_iter(:, ii, t) = [err_track_X_QR(ii)/normX; err_track_X_OLD(ii)/normX;];
    end
end

mean_Error_X_LRPR_QR = mean(tmpEr_X_LRPR_QR);
% mean_Error_X_LRPR_Newmes = mean(tmpEr_X_LRPR_Newmes);
mean_Error_X_LRPR_OLD = mean(tmpEr_X_LRPROLD);
% mean_Error_X_RWF = mean(tmpEr_X_RWF);

mean_Error_U_LRPR_QR = mean(ERULRPRQR);
% mean_Error_U_LRPR_Newmes = mean(ERULRPRmes);
mean_Error_U_LRPR_OLD = mean(TmpErULRPROld);
% mean_Error_U_RWF = mean(TmpErUrwf);

mean_Time_LRPR_QR = mean(TmpTLRPQR);
% mean_Time_LRPR_Newmes = mean(TmpTLRPmes);
mean_Time_LRPR_OLD = mean(TmpExTLRPROld);
% mean_Time_RWF = mean(TmpExTrwf);

fprintf('**************************************\n');
fprintf('Error X: ...\n');
fprintf('LRPR practical:\t\t%2.2e\n', mean_Error_X_LRPR_QR);
% fprintf('LRPR theory:\t%2.2e\n', mean_Error_X_LRPR_Newmes);
fprintf('LRPR OLD:\t\t%2.2e\n', mean_Error_X_LRPR_OLD);
% fprintf('RWF:\t\t\t%2.2e\n', mean_Error_X_RWF);
fprintf('**************************************\n');

fprintf('Error U: ... \n');
fprintf('LRPR practical:\t\t%2.2e\n', mean_Error_U_LRPR_QR);
% fprintf('LRPR theory:\t%2.2e\n', mean_Error_U_LRPR_Newmes);
fprintf('LRPR OLD:\t\t%2.2e\n', mean_Error_U_LRPR_OLD);
% fprintf('RWF:\t\t\t%2.2e\n', mean_Error_U_RWF);
fprintf('**************************************\n');
fprintf('Exe Time: ... \n');
fprintf('LRPR practical:\t\t%2.2e\n', mean_Time_LRPR_QR);
% fprintf('LRPR theory:\t%2.2e\n', mean_Time_LRPR_Newmes);
fprintf('LRPR OLD:\t\t%2.2e\n', mean_Time_LRPR_OLD);
% fprintf('RWF:\t\t\t%2.2e\n', mean_Time_RWF);

toc(tt1)


%save('data/rank_est_yes_large_mc100.mat')


final_err_SE = mean(err_SE_iter, 3);
final_err_SE_med = median(err_SE_iter, 3);
final_err_SE_std = std(err_SE_iter, 0, 3);

final_err_X = median(err_X_iter, 3);
final_err_X_std = std(err_X_iter, 0, 3);
% final_err_X_rwf = mean(err_X_rwf_iter, 2);
figure
% t1 =mean(time_QR, 2);
% t2 = mean(time_OLD, 2);
semilogy(mean(time_QR, 2), final_err_X(1, :), 'rs--', 'LineWidth', 2)
hold
semilogy(mean(time_OLD, 2), final_err_X(2, :), 'b>--', 'LineWidth', 2)
% loglog(mean(time_RWF, 2), final_err_X_rwf(1, :), 'ko--', 'LineWidth', 2)

axis tight
stry = ['mat-dist', '$$(\hat{X}^t, X))$$'];
xlabel('time', 'Fontsize', 15)
ylabel(stry, 'Interpreter', 'latex', 'Fontsize', 15)
% l1 = legend('LRPR-prac', 'LRPR-AltMin', 'RWF');
% l1 = legend('AltminLowRap', 'LRPR2');
l1 = legend('reducing mk', 'fixed mk');
set(l1, 'Fontsize', 15)
%t1 = title('m = 80, n=200, q = 400, r=4');
t1 = title('adding gaussian to m=80');
set(t1, 'Fontsize', 15)


% figure
% errorbar(mean(time_QR, 2), log10(final_err_X(1, :)), final_err_X_std(1, :),...
%     'rs--', 'LineWidth', 2);
% hold
% errorbar(mean(time_OLD, 2), log10(final_err_X(2, :)), final_err_X_std(2, :),...
%     'gs-.', 'LineWidth', 2);
% axis tight
% stry = ['mat-dist', '$$(\hat{X}^t, X))$$'];
% xlabel('time', 'Fontsize', 15)
% ylabel(stry, 'Interpreter', 'latex', 'Fontsize', 15)
% % l1 = legend('LRPR-prac', 'LRPR-AltMin', 'RWF');
% l1 = legend('AltminLowRap', 'LRPR2');
% set(l1, 'Fontsize', 15)
% t1 = title('m = 150, n=200, q = 400, r=4');
% set(t1, 'Fontsize', 15)

