%%  Attempts to speed up LRPR new algorithm

close all
clear;
clc;


%nrange = [1000];
mrange = [600]; %, 150, 200, 300, 400, 500, 600];
Params.Tmont = 1;   % Number of Monte-Carlo repeats
Err_SE_theory = zeros(length(mrange), Params.Tmont);
Err_SE_prac = zeros(length(mrange), Params.Tmont);
Err_SE_AltMin = zeros(length(mrange), Params.Tmont);
Err_SE_RWF = zeros(length(mrange), Params.Tmont);

for mm = mrange
    fprintf('m = %d', mm);
    Params.n  =  300;   % Number of rows of the low rank matrix
    Params.q  =  300;   % Number of columns of the matrix for LRPR
    Params.r  =  10;     % Rank
    Params.m = mm;     % Number of measurements
    
    Params.tnew = 10;    % Total number of main loops of new LRPR
    Params.told = 10;    % Total number of main loops of Old LRPR
    
    m_b = Params.m;          %Number of measuremnets for coefficient estimate
    m_u = Params.m;           % Number of measuremnets for subspace estimate
    m_init = 150;       % Number of measuremnets for init of subspace
    
    %Params.m  =  m_init + (m_b+m_u)*Params.tot;% Number of measurements
    
    %%~PN editing m, n, r so that the variables are globally same
    % TWF Parameters
    Paramsrwf.m  =  Params.m;% Number of measurements
    Paramsrwf.n  =  Params.n;% size of columns of coefficient matrix or x_k
    Paramsrwf.r  =  Params.r;% size of columns of coefficient matrix or b_k
    Paramsrwf.npower_iter = 50;% Number of loops for initialization of TWF with power method
    Paramsrwf.mu          = 0.2;% Parameter for gradient
    Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
    Paramsrwf.TRWF           = 25;% Number of loops for b_k with simple PR
    Paramsrwf.cplx_flag   = 0;
    % Paramstwf.alpha_y     = 3;
    % Paramstwf.alpha_h     = 5;
    % Paramstwf.alpha_ub    = 5;
    % Paramstwf.alpha_lb    = 0.3;
    % Paramstwf.grad_type   = 'TWF_Poiss';
    
    file_name = strcat(['Copmare_n', num2str(Params.n), 'm', num2str(Params.m), 'r', num2str(Params.r), 'q', num2str(Params.q)]);
    file_name_txt = strcat(file_name,'.txt');
    file_name_mat = strcat(file_name,'.mat');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Generating U and B and X
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
    TmpExTLRPROld  =   zeros(Params.told,Params.Tmont);
    
    TmpErXLRPRnew  =   zeros(Params.tnew,Params.Tmont);
    TmpErULRPRnew  =   zeros(Params.tnew,Params.Tmont);
    TmpExTLRPEnew  =   zeros(Params.tnew,Params.Tmont);
    
    % TmpErXLRPRmes  =   zeros(Params.tot,Params.Tmont);
    % TmpErULRPRmes  =   zeros(Params.tot,Params.Tmont);
    % TmpExTLRPRmes  =   zeros(Params.tot,Params.Tmont);
    %
    TmpErXLRPRqr   =   zeros(Params.tnew,Params.Tmont);
    TmpErULRPRqr   =   zeros(Params.tnew,Params.Tmont);
    TmpExTLRPRqr   =   zeros(Params.tnew,Params.Tmont);
    
    flag_lrpr_new = 0;
    
    for t = 1 : Params.Tmont
        
        fprintf('=============== Monte Carlo = %d ====================\n', t);
        
        [Ysqrt,Y,A] = Generate_Mes(X,Params,Params.m);
        
        %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        % LRPR_new
        %%%%%%%%%%%%%
        if(flag_lrpr_new)
            tic;
            [B_new, U_new] = LRPRnew(Params, Paramsrwf, Y, Ysqrt, A);
            TmpTLRPRnew(t) = toc;
            ERULRPRnew(t)  =  abs(sin(subspace(U_new, U)));
            fprintf('LRPR new error U:\t %2.2e\t\t Time:\t %2.2e\n', ERULRPRnew(t), TmpTLRPRnew(t));
        end
        %         ERRXLRPRnew= 0;
        %          for ni = 1 : Params.q
        %          ERRXLRPRnew =  ERRXLRPRnew  + min(norm(X(:,ni)-X_hat(:,ni))^2, norm(X(:,ni)+X_hat(:,ni))^2);
        %          end
        %          ERRXLRPRnew = ERRXLRPRnew / normX;
        %         fprintf('LRPRnew subspace error U:\t\t %2.2e\n', ERULRPRnew );
        
        %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        % LRPR_QR
        %%%%%%%%%%%%%
        tic;
        [B_QR, U_QR, X_hat_QR] = LRPRQR(Params, Paramsrwf, Y, Ysqrt, A);
%         B_QR = randn(Params.r, Params.q);
%         U_QR = randn(Params.n, Params.r);
%         X_hat_QR = U_QR * B_QR;
        TmpTLRPQR(t) = toc;
        ERULRPRQR(t)  =  abs(sin(subspace(U_QR, U)));
        Err_SE_prac(find(mm == mrange), t)  = ERULRPRQR(t);
        fprintf('LRPRQR error U:\t %2.2e\t\t Time:\t %2.2e\n', ERULRPRQR(t), TmpTLRPQR(t));
        %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        % LRPR_Newmes
        %%%%%%%%%%%%%
        tic;
         [B_new_sample, U_new_sample, X_new_sample] = LRPRNewmes(Params, Paramsrwf, Y, Ysqrt, A,  m_u, m_b, m_init, X);
%           B_new_sample = randn(Params.r, Params.q);
%           U_new_sample = randn(Params.n, Params.r);
%           X_new_sample = U_new_sample * B_new_sample;
        TmpTLRPmes(t) = toc;
        ERULRPRmes(t)  =  abs(sin(subspace(U_new_sample, U)));
        Err_SE_theory(find(mm == mrange), t)  = ERULRPRmes(t);
        fprintf('LRPR new sample error U:\t %2.2e\t\t Time:\t %2.2e\n', ERULRPRmes(t), TmpTLRPmes(t));

        %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        
        %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        %   LRPROLD
        %%%%%%%%%%%%%
        tic;
        [X_old, U_old]= LRPR_AltMin(Y, A, Params);
        TmpExTLRPROld(t) = toc;
        TmpErULRPROld(t) =  abs(sin(subspace(U_old, U)));
        Err_SE_AltMin(find(mm == mrange), t)  = TmpErULRPROld(t);
        fprintf('LRPR error U:\t %2.2e\t\t Time:\t %2.2e\n', TmpErULRPROld(t), TmpExTLRPROld(t));
        %         fprintf('LRTWF started!\n');
        
        %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        % RWF
        %%%%%%%%%%%%
        X_rwf  =  zeros(Params.n, Params.q);
        tic;
        for nk = 1: Params.q
            Amatrix =  A(:,:,nk)';
            A1    = @(I) Amatrix  * I;
            At    = @(Y) Amatrix' * Y;
            [x_rwf]  = RWFsimple2(Ysqrt(:,nk), Paramsrwf, A1, At);
            X_rwf(:,nk)  =  x_rwf;
        end
        [Ur,~,~] =  svd(X_rwf);
        U_rwf  =  Ur(:,1:Params.r);
        TmpExTrwf(t)    =  toc;
        TmpErUrwf(t)    =  abs(sin(subspace(U_rwf, U)));
        Err_SE_RWF(find(mm == mrange), t)  = TmpErUrwf(t);
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
            
            %   LRPR NEW
            if(flag_lrpr_new);
                x_hat       =   U_new * B_new(:, nk);
                Error_X_LRPR_new = Error_X_LRPR_new + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
            end
            
            %   LRPR QP
            %x_hat       =   U_QR * B_QR(:, nk);
            x_hat = X_hat_QR(:, nk);
            Error_X_LRPR_QR = Error_X_LRPR_QR + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
            
            %   LRPR new sample
            x_hat       =   X_new_sample(:, nk);
            Error_X_LRPR_Newmes = Error_X_LRPR_Newmes + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
            
            %   LRPR old
            x_hat       =   X_old(:, nk);
            Error_X_LRPROLD = Error_X_LRPROLD + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
            
            %   RWF
            x_hat       =   U_rwf*U_rwf'*X_rwf(:, nk);
            Error_X_RWF = Error_X_RWF + min(norm(x_opt-x_hat)^2, norm(x_opt+x_hat)^2);
            
        end
        
        if(flag_lrpr_new)
            tmpEr_X_LRPR_new(t) = Error_X_LRPR_new / normX;
        end
        tmpEr_X_LRPR_QR(t) = Error_X_LRPR_QR / normX;
        tmpEr_X_LRPR_Newmes(t) = Error_X_LRPR_Newmes / normX;
        tmpEr_X_LRPROLD(t) = Error_X_LRPROLD / normX;
        tmpEr_X_RWF(t) = Error_X_RWF / normX;
        
    end
    
    if(flag_lrpr_new)
        mean_Error_X_LRPR_new = mean(tmpEr_X_LRPR_new);
    end
    mean_Error_X_LRPR_QR = mean(tmpEr_X_LRPR_QR);
    mean_Error_X_LRPR_Newmes = mean(tmpEr_X_LRPR_Newmes);
    mean_Error_X_LRPR_OLD = mean(tmpEr_X_LRPROLD);
    mean_Error_X_RWF = mean(tmpEr_X_RWF);
    
    if(flag_lrpr_new)
        mean_Error_U_LRPR_new = mean(ERULRPRnew);
    end
    mean_Error_U_LRPR_QR = mean(ERULRPRQR);
    mean_Error_U_LRPR_Newmes = mean(ERULRPRmes);
    mean_Error_U_LRPR_OLD = mean(TmpErULRPROld);
    mean_Error_U_RWF = mean(TmpErUrwf);
    
    if(flag_lrpr_new)
        mean_Time_LRPR_new = mean(TmpTLRPRnew);
    end
    mean_Time_LRPR_QR = mean(TmpTLRPQR);
    mean_Time_LRPR_Newmes = mean(TmpTLRPmes);
    mean_Time_LRPR_OLD = mean(TmpExTLRPROld);
    mean_Time_RWF = mean(TmpExTrwf);
    
    fprintf('**************************************\n');
    fprintf('Error X: ...\n');
    %fprintf('LRPR new:\t\t%2.2e\n', mean_Error_X_LRPR_new);
    fprintf('LRPR QR:\t\t%2.2e\n', mean_Error_X_LRPR_QR);
    fprintf('LRPR new samples:\t%2.2e\n', mean_Error_X_LRPR_Newmes);
    fprintf('LRPR OLD:\t\t%2.2e\n', mean_Error_X_LRPR_OLD);
    fprintf('RWF:\t\t\t%2.2e\n', mean_Error_X_RWF);
    fprintf('**************************************\n');
    fprintf('Error U: ... \n');
    %fprintf('LRPR new:\t\t%2.2e\n', mean_Error_U_LRPR_new);
    fprintf('LRPR QR:\t\t%2.2e\n', mean_Error_U_LRPR_QR);
    fprintf('LRPR new samples:\t%2.2e\n', mean_Error_U_LRPR_Newmes);
    fprintf('LRPR OLD:\t\t%2.2e\n', mean_Error_U_LRPR_OLD);
    fprintf('RWF:\t\t\t%2.2e\n', mean_Error_U_RWF);
    fprintf('**************************************\n');
    fprintf('Exe Time: ... \n');
    %fprintf('LRPR new:\t\t%2.2e\n', mean_Time_LRPR_new);
    fprintf('LRPR QR:\t\t%2.2e\n', mean_Time_LRPR_QR);
    fprintf('LRPR new samples:\t%2.2e\n', mean_Time_LRPR_Newmes);
    fprintf('LRPR OLD:\t\t%2.2e\n', mean_Time_LRPR_OLD);
    fprintf('RWF:\t\t\t%2.2e\n', mean_Time_RWF);
end


%data_m200 = [mean_Error_U_LRPR_Newmes, mean_Error_U_LRPR_QR, mean_Error_U_LRPR_OLD, mean_Error_U_RWF];
%save('temp4.mat', 'data_m300')
