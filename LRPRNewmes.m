function [B_hat, Uo, X_hat, Uo_track] = LRPRNewmes(Params, Paramsrwf, Y, Ysqrt, A, X)


for  o = 1 :Params.tnew % Main loop
    %%%%%%%
    % Initializing the subspace
    %%%%%%%
    if o == 1
        %%computing matrix for spectral init
        [~,Y_init,Ai] = Generate_Mes(X,Params,Params.m_init);
        Yu      =   zeros(Params.n, Params.n);
        normest = 9/(Params.m_init * Params.q) * sum(Y_init(:));
        for nh = 1 : Params.q
            %normest = sqrt((13/Params.m_init) * Y(:,nh)' * Y(:, nh));
            Ytr = Y_init(:,nh) .* (abs(Y_init(:, nh)) > normest);
            Yu  =   Yu + Ai(:,:,nh) * diag(Ytr) * Ai(:,:,nh)';
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
            %fprintf('estimated rank is %d\n', Params.r);
        end
        
        [P,~,~] =   svds(Yu, Params.r);
        U_hat = P;
        Uo = U_hat;
    end
    Uo_track{o} = Uo;
    %%%%%
    
    [Ysqrt1,~,Ab] = Generate_Mes(X,Params,Params.m_b);
    B_hat  =   zeros(Params.r, Params.q);
    [Ysqrt_u,~,Au] = Generate_Mes(X,Params,Params.m_u);
    
    Chat   =   zeros(Params.m, Params.q);% Estimated phase
    %  Using Simple PR for estimating coefficients
    for ni = 1 : Params.q
        Amatrix  =  Ab(:,:,ni)' *  Uo;% Design matrices for coefficients
        A1  = @(I) Amatrix  * I;
        At  = @(Y) Amatrix' * Y;
        Paramsrwf.Tb_LRPRnew = Params.Tb_LRPRnew(o);
        [bhat] = RWFsimple(Ysqrt1(:,ni), Paramsrwf, A1, At);
        B_hat(:,ni) = bhat;
        x_k =  Uo *  B_hat(:,ni);
        Chat(:, ni) = (Au(:,:,ni)'* x_k >= 0) - (Au(:,:,ni)'* x_k < 0);
    end
    
    [Qb,~]  =  qr(B_hat');
    Bo   =   Qb(:,1:Params.r)';
    
    Zvec    =   zeros(Params.m*Params.q, 1);
    for nt = 1 : Params.q
        strt_idx    =   Params.m*(nt-1) + 1;
        end_idx     =   strt_idx + Params.m - 1;
        TempVec     =   Chat(:, nt) .* Ysqrt_u(:,nt);
        Zvec(strt_idx:end_idx, 1)   =   TempVec;
    end
    Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 1e-16, 30);
    U_hat    =   reshape(Uvec, Params.n, Params.r);
    [Qu,~]  =  qr(U_hat);
    Uo  =  Qu(:, 1:Params.r);
    X_hat = Uo * B_hat;
end

    function x_out = mult_H(x_in)
        X_mat    =   reshape(x_in, Params.n, Params.r);
        x_out    =   zeros(Params.q*Params.m, 1);
        for na = 1: Params.q
            x_out((na-1)*Params.m + 1 : na*Params.m) = Au(:,:,na)' * X_mat * Bo(:,na);
        end
    end

    function w_out = mult_Ht(w_in)
        w_out   =   zeros(Params.n*Params.r, 1);
        for na = 1: Params.q
            tmp_vec  =   Au(:,:,na) * w_in((na-1)*Params.m+1:na*Params.m);
            w_out    =   w_out + kron(Bo(:,na), tmp_vec);
        end
    end
end