tt1 = tic;
Params.Tmont = 1;

Params.n  =  200;   % Number of rows of the low rank matrix
Params.q  =  150;   % Number of columns of the matrix for LRPR
Params.r  =  2;     % Rank
Params.m = Params.n/4;     % Number of measurements

Params.tnew = 10;    % Total number of main loops of new LRPR
Params.told = 10;    % Total number of main loops of Old LRPR

Params.m_b = Params.m;          %Number of measuremnets for coefficient estimate
Params.m_u = Params.m;           % Number of measuremnets for subspace estimate
Params.m_init = Params.m;       % Number of measuremnets for init of subspace
%m_init = 50;
Params.rank_est_flag = 1;
Paramsrwf.r = Params.r;
Paramsrwf.proj =1;
%Params.m  =  m_init + (m_b+m_u)*Params.tot;% Number of measurements

%%~PN editing m, n, r so that the variables are globally same
% TWF Parameters
Paramsrwf.m  =  ceil(5 * Params.n);% Number of measurements
Paramsrwf.n  =  Params.n;% size of columns of coefficient matrix or x_k
Paramsrwf.r  =  Params.r;% size of columns of coefficient matrix or b_k
Paramsrwf.npower_iter = 30;% Number of loops for initialization of TWF with power method
Paramsrwf.mu          = 0.2;% Parameter for gradient
Params.Tb_LRPRnew    = unique(ceil(linspace(5, 20, Params.tnew)));% Number of loops for b_k with simple PR
%Params.Tb_LRPRnew    = 85 * ones(1, Params.tnew);
% Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
Paramsrwf.TRWF           = 300;% Number of loops for b_k with simple PR
err_rwf = zeros(Paramsrwf.TRWF + 1, Params.q);
err_rwf2 = zeros(Paramsrwf.TRWF + 1, Params.q);

Paramsrwf.cplx_flag   = 0;
Paramstwf.cplx_flag   = 0;
Paramstwf.alpha_y     = 3;
Paramstwf.alpha_h     = 5;
Paramstwf.alpha_ub    = 5;
Paramstwf.alpha_lb    = 0.3;
Paramstwf.grad_type   = 'TWF_Poiss';
Paramstwf.npower_iter = 30;
Paramstwf.n = Params.n;
Paramstwf.mu = 0.2;
Paramstwf.T = 300;
Paramstwf.m = Params.m;
err_twf = zeros(Paramstwf.T, Params.q);
%Params.seed = rng;
err_SE_iter = zeros(2, Params.tnew, Params.Tmont);
err_X_iter = zeros(2, Params.tnew, Params.Tmont);
err_X_rwf_iter = zeros(Paramsrwf.TRWF+1, Params.Tmont);
err_X_rwf_iter2 = zeros(Paramsrwf.TRWF+1, Params.Tmont);
err_X_twf_iter = zeros(Paramstwf.T, Params.Tmont);

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

Params.sig_star = svds(X, Params.r);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Compare


time_QR = zeros(Params.tnew, Params.Tmont);
time_OLD = zeros(Params.told, Params.Tmont);
time_RWF = zeros(Paramsrwf.TRWF, Params.Tmont);
time_RWF2 = zeros(Paramsrwf.TRWF, Params.Tmont);
time_TWF = zeros(Paramstwf.T, Params.Tmont);

for t = 1 : Params.Tmont
    [Ysqrt,Y,A] = Generate_Mes(X,Params,Params.m);
    fprintf('=============== Monte Carlo = %d ====================\n', t);
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % LRPR - practice
    %%%%%%%%%%%%%
    tic;
    [B_QR, U_QR, X_hat_QR, U_track_QR, X_track_QR, time_QR(:, t)] = LRPRQR(Params, Paramsrwf, Y, Ysqrt, A);
    fprintf('LRPR-practice error U:\t %2.2e\t\t Time:\t %2.2e\n', ERULRPRQR(t), TmpTLRPQR(t));
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % RWF
    %%%%%%%%%%%%
    X_rwf  =  zeros(Params.n, Params.q);
    X_twf  =  zeros(Params.n, Params.q);
    tic;
    for nk = 1: Params.q
        Amatrix =  A(:,:,nk)';
        A1    = @(I) Amatrix  * I;
        At    = @(Y) Amatrix' * Y;
         [x_rwf, err_rwf(:, nk), time_temp]  = RWFsimple2(Ysqrt(:,nk), Paramsrwf, A1, At, X(:, nk));
         time_RWF(:, t) = time_RWF(:, t) + time_temp;
         
[x_twf, err_twf(:, nk), time_temp]  = TWF(Y(:,nk), Paramstwf, A1, At, X(:,nk));
time_RWF(:, t) = time_RWF(:, t) + time_temp';
        X_rwf(:,nk)  =  x_rwf;
        X_twf(:,nk) = x_twf;
    end
    
    
    
    err_X_rwf_iter(:, t) = sum(err_rwf, 2)/normX;
    [Ur,~,~] =  svd(X_rwf);
    U_rwf  =  Ur(:,1:Params.r);
    TmpExTrwf(t)    =  toc;
    TmpErUrwf(t)    =  abs(sin(subspace(U_rwf, U)));
    fprintf('RWF subspace error:\t%2.2e\t\tTime: %2.2e\n', TmpErUrwf(t), TmpExTrwf(t));
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % Error X
    %%%%%%%%%%%%
    Error_X_LRPR_new = 0;
    Error_X_LRPR_QR = 0;
    Error_X_LRPR_Newmes = 0;
    Error_X_LRPROLD = 0;
    Error_X_RWF = 0;
    
    for nk = 1 : Params.q
        x_opt       =   X(:, nk);
        
        %   LRPR practical
%         x_hat = X_hat_QR(:, nk);
%         Error_X_LRPR_QR = Error_X_LRPR_QR + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
        
        %   LRPR theory
        %         x_hat       =   X_new_sample(:, nk);
        %         Error_X_LRPR_Newmes = Error_X_LRPR_Newmes + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
        
        %   LRPR old
%         x_hat       =   X_old(:, nk);
%         Error_X_LRPROLD = Error_X_LRPROLD + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
        
          %RWF
        x_hat       =   U_rwf*U_rwf'*X_rwf(:, nk);
        Error_X_RWF = Error_X_RWF + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
    end
    tmpEr_X_LRPR_QR(t) = Error_X_LRPR_QR / normX;
    %     tmpEr_X_LRPR_Newmes(t) = Error_X_LRPR_Newmes / normX;
    tmpEr_X_LRPROLD(t) = Error_X_LRPROLD / normX;
    tmpEr_X_RWF(t) = Error_X_RWF / normX;
%         for ii = 1 : Params.tnew
%             err_SE_iter(:, ii, t) = [abs(sin(subspace(U_track_QR{ii}, U))); ...
%                 abs(sin(subspace(U_track_new{ii}, U))); ...
%                 abs(sin(subspace(U_track_old{ii}, U)));];
%         end
%     
    err_track_X_QR = zeros(Params.tnew,1);
    err_track_X_OLD = zeros(Params.tnew,1);
    
    for ii = 1 : Params.tnew
%         err_SE_iter(:, ii, t) = [abs(sin(subspace(U_track_QR{ii}, U))); ...
%             abs(sin(subspace(U_track_old{ii}, U)));];
        for jj = 1 : Params.q
%             x_opt       =   X(:, nk);
%             
%             XhatQR = X_track_QR{ii};
%             x_hat = XhatQR(:, nk);
%             err_track_X_QR(ii) = err_track_X_QR(ii) + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
%             
%             XhatOLD = X_track_OLD{ii};
%             x_hat       =   XhatOLD(:, nk);
%             err_track_X_OLD(ii) = err_track_X_OLD(ii) + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
            
        end
        err_track_X_OLD(ii) = err_track_X_OLD(ii)/Params.q;
        err_track_X_QR(ii) = err_track_X_QR(ii)/Params.q;
        
        err_X_iter(:, ii, t) = [err_track_X_QR(ii)/normX; err_track_X_OLD(ii)/normX;];
    end
end


toc(tt1)




time_RWF1 = time_RWF;


final_err_X = median(err_X_iter, 3);

final_err_X_rwf1 = mean(err_X_rwf_iter, 2);
final_err_X_rwf1 = min(err_X_rwf_iter(:,1)) * ones(301,1);
final_err_X_rwf1 = median(err_X_rwf_iter, 2);
final_err_X_rwf_std = median(err_X_rwf_iter, 2);

figure
 semilogy(mean(time_RWF1, 2), final_err_X_rwf1,  'ko--', 'LineWidth', 2)
%semilogy(mean(time_RWF1, 2), final_err_X_rwf1(1:end-1),  'ko--', 'LineWidth', 2)

figure
% t1 =mean(time_QR, 2);
% t2 = mean(time_OLD, 2);
loglog(lrpr_suc_time3, lrpr_suc_err3, 'rs--', 'LineWidth', 2)
hold
loglog(rwf_suc_time1(1:50), rwf_suc_err1(1:50), 'b>--', 'LineWidth', 2)
loglog(rwf_fail_time1, rwf_fail_err1,  'gd--', 'LineWidth', 2)
loglog(twf_suc_time1(1:50), twf_suc_err1(1:50),  'ko--', 'LineWidth', 2)
axis tight
stry = ['mat-dist', '$$(\hat{X}^t, X)$$'];
xlabel('time', 'Fontsize', 15)
ylabel(stry, 'Interpreter', 'latex', 'Fontsize', 15)
l1 = legend('AltminLowRap m = n/4', 'RWF m = 5n', 'RWF m=4n', 'TWF m=5n');
set(l1, 'Fontsize', 15)
t1 = title('n = 200, q = 150, r=2');
% set(t1, 'Fontsize', 15)

