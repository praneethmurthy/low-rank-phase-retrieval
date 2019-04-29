%%  Attempts to speed up LRPR new algorithm

%close all
clear;
clc;


tt1 = tic;
Params.Tmont = 1;

Params.n  =  600;   % Number of rows of the low rank matrix
Params.q  =  1000;   % Number of columns of the matrix for LRPR
Params.r  =  4;     % Rank
Params.m = 150;     % Number of measurements

Params.tnew = 10;    % Total number of main loops of new LRPR
Params.told = 10;    % Total number of main loops of Old LRPR

Params.m_b = Params.m;          %Number of measuremnets for coefficient estimate
Params.m_u = Params.m;           % Number of measuremnets for subspace estimate
Params.m_init = Params.m;       % Number of measuremnets for init of subspace
%m_init = 50;
Params.rank_est_flag = 1;
Paramsrwf.r = Params.r;
Paramsrwf.proj =0;
%Params.m  =  m_init + (m_b+m_u)*Params.tot;% Number of measurements

%%~PN editing m, n, r so that the variables are globally same
% TWF Parameters
Paramsrwf.m  =  Params.m;% Number of measurements
Paramsrwf.n  =  Params.n;% size of columns of coefficient matrix or x_k
Paramsrwf.r  =  Params.r;% size of columns of coefficient matrix or b_k
Paramsrwf.npower_iter = 300;% Number of loops for initialization of TWF with power method
Paramsrwf.mu          = 0.2;% Parameter for gradient
Params.Tb_LRPRnew    = unique(ceil(linspace(5, 20, Params.tnew)));% Number of loops for b_k with simple PR
%Params.Tb_LRPRnew    = 85 * ones(1, Params.tnew);
% Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
Paramsrwf.TRWF           = 100;% Number of loops for b_k with simple PR
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
% time_RWF = zeros(Paramsrwf.TRWF, Params.Tmont);
time_RWF = zeros(Paramsrwf.TRWF+1,1);

for t = 1 : Params.Tmont
    [Ysqrt,Y,A] = Generate_Mes(X,Params,Params.m);
    fprintf('=============== Monte Carlo = %d ====================\n', t);
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % LRPR - practice
    %%%%%%%%%%%%%
%     tic;
%     [B_QR, U_QR, X_hat_QR, U_track_QR, X_track_QR, time_QR(:, t)] = LRPRQR(Params, Paramsrwf, Y, Ysqrt, A);
%     TmpTLRPQR(t) = toc;
%     ERULRPRQR(t)  =  abs(sin(subspace(U_QR, U)));
%     fprintf('LRPR-practice error U:\t %2.2e\t\t Time:\t %2.2e\n', ERULRPRQR(t), TmpTLRPQR(t));
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % LRPR - theory
    %%%%%%%%%%%%%
    %     tic;
    %     [B_new_sample, U_new_sample, X_new_sample, U_track_new] = ...
    %         LRPRNewmes(Params, Paramsrwf, Y, Ysqrt, A, X);
    %     TmpTLRPmes(t) = toc;
    %     ERULRPRmes(t)  =  abs(sin(subspace(U_new_sample, U)));
    %     fprintf('LRPR theory error U:\t %2.2e\t\t Time:\t %2.2e\n', ERULRPRmes(t), TmpTLRPmes(t));
    
%     TmpTLRPmes(t) = 0;
%     ERULRPRmes(t)  =  eps;
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    %   LRPR Alt Min
    %%%%%%%%%%%%%
%     tic;
%     [X_old, U_old, U_track_old, X_track_OLD, time_OLD(:, t)]= LRPR_AltMin(Y, A, Params);
%     TmpExTLRPROld(t) = toc;
%     TmpErULRPROld(t) =  abs(sin(subspace(U_old, U)));
%     fprintf('LRPR error U:\t %2.2e\t\t Time:\t %2.2e\n', TmpErULRPROld(t), TmpExTLRPROld(t));
    
    %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    % RWF
    %%%%%%%%%%%%
    X_rwf  =  zeros(Params.n, Params.q);
    tt_rwf = tic;
    npower_iter = Paramsrwf.npower_iter;
    tt=1;
    while tt<=Paramsrwf.TRWF+1
        if (tt==1)
            for nk = 1: Params.q
                y1 = Ysqrt(:, nk);
                Amatrix =  A(:,:,nk)';
                A1    = @(I) Amatrix  * I;
                At    = @(Y) Amatrix' * Y;
                z0 = randn(Params.n, 1); z0 = z0/norm(z0,'fro');    
                normest = (sqrt(pi/2)*(1-Paramsrwf.cplx_flag) ...
                    +sqrt(4/pi)*Paramsrwf.cplx_flag)*sum(y1(:))/numel(y1(:));
                % Estimate norm to scale eigenvector
                
                ytr=y1.* (abs(y1) > 1 * normest );% truncated version
                for nn = 1: npower_iter
                    z0 = At( ytr.* (A1(z0)) ); z0 = z0/norm(z0,'fro');
                end
                z0 = normest * z0; % Apply scaling
                z = z0;
                X_rwf(:, nk) = z0;
                err_rwf(tt, nk)=norm(X(:,nk) - exp(-1i*angle(trace(X(:,nk)'*z)))...
                    * z, 'fro')/norm(X(:,nk),'fro');
            end
            
            if (Paramsrwf.proj ==1)
                [u, ~] = svds(X_rwf, Params.r);
                X_rwf = u * (u' * X_rwf);
            end
            %% reshaped Wirtinger flow
            %Relerrs=zeros(Params.T+1,1);
            z0 = X_rwf;
            z=z0;
            time_RWF(tt) = toc(tt_rwf);
        end
        
        mu=0.8+0.4*Paramsrwf.cplx_flag;
        for nk = 1 : Params.q
            y1 = Ysqrt(:, nk);
            Amatrix =  A(:,:,nk)';
            z = X_rwf(:, nk); 
            A1    = @(I) Amatrix  * I;
            At    = @(Y) Amatrix' * Y;
            yz=A1(z);
            %   ang   =  Params.cplx_flag*exp(1i * angle(yz)) +(1 - Params.cplx_flag) * sign(yz);
            z = z - mu* (Params.m\At(yz-y1.*yz./abs(yz)));
            X_rwf(:, nk) = z;
            err_rwf(tt, nk)=norm(X(:,nk) - exp(-1i*angle(trace(X(:,nk)'*z)))...
                    * z, 'fro')/norm(X(:,nk),'fro');
        end
        if(Paramsrwf.proj ==1)
            [u, ~] = svds(X_rwf, Params.r);
            X_rwf = u * (u' * X_rwf);
        end
        time_RWF(tt) = toc(tt_rwf);
        %  Relerrs(t+1)=norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro');
        tt=tt+1;
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

mean_Error_X_LRPR_QR = mean(tmpEr_X_LRPR_QR);
% mean_Error_X_LRPR_Newmes = mean(tmpEr_X_LRPR_Newmes);
mean_Error_X_LRPR_OLD = mean(tmpEr_X_LRPROLD);
mean_Error_X_RWF = mean(tmpEr_X_RWF);

% mean_Error_U_LRPR_QR = mean(ERULRPRQR);
% mean_Error_U_LRPR_Newmes = mean(ERULRPRmes);
% mean_Error_U_LRPR_OLD = mean(TmpErULRPROld);
mean_Error_U_RWF = mean(TmpErUrwf);

% mean_Time_LRPR_QR = mean(TmpTLRPQR);
% mean_Time_LRPR_Newmes = mean(TmpTLRPmes);
% mean_Time_LRPR_OLD = mean(TmpExTLRPROld);
mean_Time_RWF = mean(TmpExTrwf);


toc(tt1)


time_RWF1 = time_RWF;


final_err_X = mean(err_X_iter, 3);

final_err_X_rwf1 = mean(err_X_rwf_iter, 2);
final_err_X_rwf1 = min(err_X_rwf_iter(:,1)) * ones(101,1);
final_err_X_rwf2 = median(err_X_rwf_iter, 2);
time_RWF2 = time_RWF1;
final_err_X_rwf_std = median(err_X_rwf_iter, 2);
figure
% t1 =mean(time_QR, 2);
% t2 = mean(time_OLD, 2);
% semilogy(mean(time_QR, 2), final_err_X(1, :), 'rs--', 'LineWidth', 2)
% hold
% semilogy(mean(time_OLD, 2), final_err_X(2, :), 'b>--', 'LineWidth', 2)
% semilogy(mean(time_RWF1, 2), final_err_X_rwf1(1, :),  'ko--', 'LineWidth', 2)
semilogy(time_RWF1, final_err_X_rwf1,  'ko--', 'LineWidth', 2)
axis tight
stry = ['log(mat-dist', '$$(\hat{X}^t, X))$$'];
xlabel('time', 'Fontsize', 15)
ylabel(stry, 'Interpreter', 'latex', 'Fontsize', 15)
% l1 = legend('LRPR-prac', 'LRPR-AltMin', 'RWF');
l1 = legend('Proj RWF');
set(l1, 'Fontsize', 15)
t1 = title('m = 150, n = 600, q = 1000, r = 4');
set(t1, 'Fontsize', 15)

% figure
% errorbar(mean(time_QR, 2), log10(final_err_X(1, :)), log10(final_err_X_std(1, :)),...
%     'rs--', 'LineWidth', 2);
% hold
% errorbar(mean(time_OLD, 2), log10(final_err_X(2, :)), final_err_X_std(2, :),...
%     'b>--', 'LineWidth', 2);
% errorbar(mean(time_RWF1, 2), log10(min(final_err_X_rwf1(1:end-1)) * ones(300,1)), eps * ones(300,1), ...
%     'ko--', 'LineWidth', 2)
% axis tight
% stry = ['log(mat-dist', '$$(\hat{X}^t, X))$$'];
% xlabel('time', 'Fontsize', 15)
% ylabel(stry, 'Interpreter', 'latex', 'Fontsize', 15)
% l1 = legend('LRPR-prac', 'LRPR-AltMin', 'RWF');
% l1 = legend('AltMin-PhaPCA', 'LRPR2', 'RWF');
% set(l1, 'Fontsize', 15)
% t1 = title('m = 80, n = 200, q = 400, r=4');
% set(t1, 'Fontsize', 15)

