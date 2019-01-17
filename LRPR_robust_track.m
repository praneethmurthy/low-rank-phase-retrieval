function [B_hat, Uo, Uo_track, t_calc] = LRPR_robust_track(Params, Paramsrwf, Y, Ysqrt, A, X)
[~,k_max] = size(Y);
det_mode = 0; %0  means not-detect and 1 means detect
khat = [1];
l=0;
ctr = 1;
t_calc = [];
% %need to define alpha, L, thresh
for kk = 2 : k_max
    if((~mod(kk-1, Params.alpha)) &&(kk == khat(end) + Params.alpha)) %% this means alpha frames after previous dectection
        [~,Y_init,Ai] = ...
            Generate_Mes_temp(X(:, kk - Params.alpha:kk), Params,Params.m_init, Params.alpha);
        Yu      =   zeros(Params.n, Params.n);
        for nh = 1 : Params.alpha
            normest = sqrt((9/Params.m) * Y_init(:,nh)' * Y_init(:, nh));
            Ytr = Y_init(:,nh) .* (abs(Y_init(:, nh)) > normest);
            Yu  =   Yu + Ai(:,:,nh) * diag(Ytr) * Ai(:,:,nh)';
        end
        Yu      =   Yu / Params.q / Params.m;
        [U_hat,~,~] =   svds(Yu, Params.r);
        Uo = U_hat;
        Uo_track{ctr} = Uo;
        t_calc = [t_calc, kk];
        ctr = ctr + 1;
        %%estimate the coefficients bhat's
        [Ysqrt1,~,Ab] = ...
            Generate_Mes_temp(X(:, kk - Params.alpha:kk), Params,Params.m_u, Params.alpha);
        [~,~,Au] = ...
            Generate_Mes_temp(X(:, kk - Params.alpha:kk), Params,Params.m_b, Params.alpha);
        B_hat  =   zeros(Params.r, Params.alpha);
        Chat   =   zeros(Params.m, Params.alpha);
        for ni = 1 : Params.alpha
            Amatrix  =  Ab(:,:,ni)' *  Uo;% Design matrices for coefficients
            A1  = @(I) Amatrix  * I;
            At  = @(Y) Amatrix' * Y;
            Paramsrwf.Tb_LRPRnew = Params.Tb_LRPRnew(1); %%hacky soln -- T_a iterations of RWF!
            [bhat] = RWFsimple(Ysqrt1(:,ni), Paramsrwf, A1, At);
            B_hat(:,ni) = bhat;
            x_k =  Uo *  B_hat(:,ni);
            Chat(:, ni) = (Au(:,:,ni)'* x_k >= 0) - (Au(:,:,ni)'* x_k < 0);
        end
        fprintf('subspace initialized at %d and ctr is %d\n', kk, ctr);
        
    elseif((~mod(kk-1, Params.alpha)) && (kk ~= khat(end) + Params.alpha)) %%not initializing
        l = l + 1;
        [Ysqrt1,~,Ab] = ...
            Generate_Mes_temp(X(:, kk - Params.alpha:kk), Params,Params.m_b, Params.alpha);
        [Ysqrt_u,~,Au] = ...
            Generate_Mes_temp(X(:, kk - Params.alpha:kk), Params,Params.m_u, Params.alpha);
        B_hat  =   zeros(Params.r, Params.alpha);
        Chat   =   zeros(Params.m, Params.alpha);
        for ni = 1 : Params.alpha
            Amatrix  =  Ab(:,:,ni)' *  Uo;% Design matrices for coefficients
            A1  = @(I) Amatrix  * I;
            At  = @(Y) Amatrix' * Y;
            Paramsrwf.Tb_LRPRnew = Params.Tb_LRPRnew(1); %%hacky soln -- T_a iterations of RWF!
            [bhat] = RWFsimple(Ysqrt1(:,ni), Paramsrwf, A1, At);
            B_hat(:,ni) = bhat;
            x_k =  Uo *  B_hat(:,ni);
            Chat(:, ni) = (Au(:,:,ni)'* x_k >= 0) - (Au(:,:,ni)'* x_k < 0);
        end
        [Qb,~]  =  qr(B_hat');
        Bo   =   Qb(:,1:Params.r)';
        
        %[Ysqrt_u,~,Au] = Generate_Mes(X,Params,Params.m_u);
        Zvec    =   zeros(Params.m*Params.alpha, 1);
        for nt = 1 : Params.alpha
            strt_idx    =   Params.m*(nt-1) + 1;
            end_idx     =   strt_idx + Params.m - 1;
            TempVec     =   Chat(:, nt) .* Ysqrt_u(:,nt);
            Zvec(strt_idx:end_idx, 1)   =   TempVec;
        end
        Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 1e-16,30);
        U_hat    =   reshape(Uvec, Params.n, Params.r);
        [Qu,~]  =  qr(U_hat);
        Uo  =  Qu(:, 1:Params.r);
        Uo_track{ctr} = Uo;
        t_calc = [t_calc, kk];
        ctr = ctr+1;
        fprintf('subspace updated at %d and ctr is %d\n', kk, ctr);
        %X_hat = Uo * B_hat;
    end
    
    if((~mod(kk-1, Params.alpha)) &&(l >= Params.L)) %%means epsilon accurate subspace is obtained and need to check for ss change
        fprintf('checking for change at %d\n', kk)
        det_mode = 1;
    end
    
    if((~mod(kk-1, Params.alpha)) && (det_mode == 1))
        [~,Y_init,Ai] = ...
            Generate_Mes_temp(X(:, kk - Params.alpha:kk), Params,Params.m_init, Params.alpha);
        Yu      =   zeros(Params.n, Params.n);
        for nh = 1 : Params.alpha
            normest = sqrt((9/Params.m) * Y_init(:,nh)' * Y_init(:, nh));
            Ytr = Y_init(:,nh) .* (abs(Y_init(:, nh)) > normest);
            Yu  =   Yu + Ai(:,:,nh) * diag(Ytr) * Ai(:,:,nh)';
        end
        Yu      =   Yu / Params.q / Params.m;
        Y_u_det = (Yu - (Uo * (Uo' * Yu)));
        [~,sig_det] = svds(Y_u_det);
        fprintf('the detection criterion value at %d is %f\n',kk, max(sig_det(:)) - min(sig_det(:)));
        if(max(sig_det(:)) - min(sig_det(:)) >= Params.thresh)
            det_mode = 0;
            l = 0;
            khat = [khat, kk];
            fprintf('change detected at %d\n', kk);
            fprintf('Max: %f \t Min: %f\n', max(sig_det(:)), min(sig_det(:)));
        end
    end
end

    function x_out = mult_H(x_in)
        X_mat    =   reshape(x_in, Params.n, Params.r);
        x_out    =   zeros(Params.alpha*Params.m, 1);
        for na = 1: Params.alpha
            x_out((na-1)*Params.m + 1 : na*Params.m) = Au(:,:,na)' * X_mat * Bo(:,na);
        end
    end

    function w_out = mult_Ht(w_in)
        w_out   =   zeros(Params.n*Params.r, 1);
        for na = 1: Params.alpha
            tmp_vec  =   Au(:,:,na) * w_in((na-1)*Params.m+1:na*Params.m);
            w_out    =   w_out + kron(Bo(:,na), tmp_vec);
        end
    end
end