function [B_hat, Uo, X_hat, Uo_track, X_track, time_iter] = LRPRQR(Params, Paramsrwf, Y, Ysqrt, A)
time_iter = zeros(Params.tnew, 1);
tic;
for  o = 1 :Params.tnew % Main loop
    %%%%%%%
    % Initializing the subspace
    %%%%%%%
    if o == 1
        %%computing matrix for spectral init
        %[~,Y_init,Ai] = Generate_Mes(X,Params,Params.m_init);
        Yu      =   zeros(Params.n, Params.n);
        normest = 9/(Params.m * Params.q) * sum(Y(:));
        for nh = 1 : Params.q
            %normest = sqrt((13/Params.m_init) * Y(:,nh)' * Y(:, nh));
            Ytr = Y(:,nh) .* (abs(Y(:, nh)) > normest);
            Yu  =   Yu + A(:,:,nh) * diag(Ytr) * A(:,:,nh)';
        end
        Yu      =   Yu / Params.q / Params.m_init;

        if(Params.rank_est_flag == 1) %%need to estimate rank
            %%checking rank estimation
            [~,sig_init,~] =   svd(Yu);
            sig_init = diag(sig_init);
            tmp1 = 1.* (sig_init(1:end-1) - min(sig_init) >= 1.3 * min(Params.sig_star)^2/Params.q);
            if (all(tmp1 == 0))
                est_rank = 1;
            else
                est_rank = find(tmp1 == 1, 1, 'last');
            end
            Params.r = est_rank;
            Paramsrwf.r  =  Params.r;
            fprintf('estimated rank is %d\n', Params.r);
        end
        
        [P,~,~] =   svds(Yu, Params.r);
        U_hat = P;
        Uo = U_hat;
    end
    Uo_track{o} = Uo;
    
    %%%%%
    
    X_hat = zeros(Params.n, Params.q);
    B_hat  =   zeros(Params.r, Params.q);
    Chat   =   zeros(Params.m, Params.q);% Estimated phase
    %  Using Simple PR for estimating coefficients
    for ni = 1 : Params.q
        Amatrix  =  A(:,:,ni)' *  Uo;% Design matrices for coefficients
        A1  = @(I) Amatrix  * I;
        At  = @(Y) Amatrix' * Y;
        %atrue = B(:,ni);
        %[a, Relerrs] = TWFsimple(Y(:,ni),atrue, Paramstwf, A1, At);
        Paramsrwf.Tb_LRPRnew = Params.Tb_LRPRnew(o);
        [bhat] = RWFsimple(Ysqrt(:,ni), Paramsrwf, A1, At);
        B_hat(:,ni) = bhat;
        x_k =  Uo *  B_hat(:,ni);
        Chat(:,ni) = (A(:,:,ni)'* x_k >= 0) - (A(:,:,ni)'* x_k < 0);
        X_hat(:, ni) = x_k;
    end
    time_iter(o) = toc;
    X_track{o} = Uo * B_hat;
    [Qb,~]  =  qr(B_hat');
    Bo   =   Qb(:,1:Params.r)';
    
    
    % Estimating the subspace
    Zvec    =   zeros(Params.m*Params.q, 1);
    %tic;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       Updating Uhat
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for nt = 1 : Params.q
        strt_idx    =   Params.m*(nt-1) + 1;
        end_idx     =   strt_idx + Params.m - 1;
        TempVec     =   Chat(:,nt) .* Ysqrt(:,nt);
        Zvec(strt_idx:end_idx, 1)   =   TempVec;
    end
    Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 1e-16, 30);
    U_hat    =   reshape(Uvec, Params.n, Params.r);
    [Qu,~]  =  qr(U_hat);
    Uo  =  Qu(:, 1:Params.r);
end
    function x_out = mult_H(x_in)
        X_mat    =   reshape(x_in, Params.n, Params.r);
        %    x_out    =   A_long * X_vec;
        x_out    =   zeros(Params.q*Params.m, 1);
        for na = 1: Params.q
            x_out((na-1)*Params.m + 1 : na*Params.m) = A(:,:,na)' * X_mat * Bo(:,na);
        end
        
    end


    function w_out = mult_Ht(w_in)
        
        w_out   =   zeros(Params.n*Params.r, 1);
        for na = 1: Params.q
            tmp_vec  =   A(:,:,na) * w_in((na-1)*Params.m+1:na*Params.m);
            w_out    =   w_out + kron(Bo(:,na), tmp_vec);
        end
        
    end
end