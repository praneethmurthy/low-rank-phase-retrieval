function [B_hat, Uo, X_hat, Uo_track] = ...
    LRPR_prac_video(Params, Paramsrwf, Y, Afull, Afull_t, Afull_tk, Masks2)
Ysqrt = sqrt(Y);
%%contains some changes to consider the CDP setting. Check if it can be
%%made general to be able to handle simulated data too

for  o = 1 :Params.tnew % Main loop
    %%%%%%%
    % Initialization
    %%%%%%%
    if o == 1
        Ytrk    =   zeros(Params.n_1,Params.n_2,Params.L,Params.q);
        for ni = 1 : Params.q
            Yk  =   reshape(Y(:,:,:,ni), Params.n_1*Params.n_2*Params.L, 1);
            normest =   sum(Yk(:))/Params.m;
            Eyk     =   (Yk <= Params.alpha_y^2*normest);
            % num_larg = sum(Eyk);
            Ytrk(:,:,:,ni)  =   reshape(Eyk.*Yk,Params.n_1,Params.n_2,Params.L);
        end
        % Block power method for estimating initial U
        U_tmp1  = randn(Params.n_1*Params.n_2,Params.r);
        [U_upd_vec, ~, ~]   =   qr(U_tmp1, 0);
        Uupdt          =   reshape(U_upd_vec, Params.n_1, Params.n_2, Params.r);
        U_tmp   =  zeros(Params.n_1,Params.n_2,Params.r) ;
        
        for t = 1 : Params.itr_num_pow_mth
            fprintf('power method iteration %d\n', t);
            for nr  =   1 : Params.r
                U_tmp(:,:,nr) = Afull_t( Ytrk.* Afull(repmat(Uupdt(:,:,nr), [1,1,Params.q])));
            end
            [Uupdt3, ~, ~]   =   qr(reshape(U_tmp, Params.n_1*Params.n_2, Params.r), 0);
            Uupdt          =   reshape(Uupdt3, Params.n_1, Params.n_2, Params.r);
        end
        Uhat_vec    =   reshape(Uupdt, Params.n_1*Params.n_2, Params.r);
        Uhat            =    Uhat_vec;
        %[Qu,~] = qr(Uhat);
        
        Uo = Uhat; %Qu(:, 1 : Params.r);
    end
    
    AUnew   =   zeros(Params.n_1*Params.n_2*Params.L,Params.q,Params.r);
    Ybk     =   zeros(Params.n_1,Params.n_2,Params.r,Params.q);
    for nr  =   1 : Params.r
        AUnew(:,:,nr) =  reshape(Afull(repmat(Uupdt(:,:,nr),[1,1,Params.q])), Params.n_1*Params.n_2*Params.L,Params.q);
        Ybk(:,:,nr,:) =  Afull_tk(Y.* Afull(repmat(Uupdt(:,:,nr), [1,1,Params.q])));
    end
    
    B_hat  =   zeros(Params.n_1, Params.n_2, Params.q);
    
%     for na = 1 : Params.q
%         ybk = Uhat_vec' * reshape(Ybk(:,:,:,na),Params.n_1*Params.n_2,Params.r);
%         [b_t1,~,~]   =   svd(ybk);
%         nb =   sqrt(sum(reshape(Y(:,:,:,na),Params.n_1*Params.n_2*Params.L,1))/Params.m);
%         nc = Params.m*Params.r/(norm(reshape(AUnew(:,na,:),Params.n_1*Params.n_2*Params.L,Params.r),'fro')^2);
%         norm_b_a_hat1=   sqrt(nc)* nb;
%         Bhat(:,na)   =   norm_b_a_hat1 * b_t1(:,1);
%     end
    
    %Chat   =   zeros(Params.m, Params.q);% Estimated phase
    %  Using Simple PR for estimating coefficients
    for ni = 1 : Params.q
        %Amatrix  =  A(:,:,ni)' *  Uo;% Design matrices for coefficients
        %A1  = @(I) Amatrix  * I;
        %At  = @(Y) Amatrix' * Y;
        %atrue = B(:,ni);
        %[a, Relerrs] = TWFsimple(Y(:,ni),atrue, Paramstwf, A1, At);
        Masks  =   Masks2(:,:,:,ni);
        A_pr  = @(I)  fft2(conj(Masks) .* reshape(repmat(I,[1 Params.L]), size(I,1), size(I,2), Params.L));
        At_pr = @(W) sum(Masks .* ifft2(W), 3) * size(W,1) * size(W,2);

        Paramsrwf.Tb_LRPRnew = Params.Tb_LRPRnew(o);
        [bhat] = RWF_2d(sqrt(mean(Y(:, :, :,ni), 3)), Paramsrwf, A_pr, At_pr);
        B_hat(:, :,ni) = bhat;
        x_k =  Uo *  B_hat(:,ni);
        Chat(:,ni) = (At_pr(x_k) >= 0) - (At_pr(x_k) < 0);
        X_hat(:, ni) = x_k;
    end
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
            x_out((na-1)*Params.m + 1 : na*Params.m) = Afull(:,:,na)' * X_mat * Bo(:,na);
        end
        
    end


    function w_out = mult_Ht(w_in)
        
        w_out   =   zeros(Params.n*Params.r, 1);
        for na = 1: Params.q
            tmp_vec  =   Afull(:,:,na) * w_in((na-1)*Params.m+1:na*Params.m);
            w_out    =   w_out + kron(Bo(:,na), tmp_vec);
        end
        
    end
end