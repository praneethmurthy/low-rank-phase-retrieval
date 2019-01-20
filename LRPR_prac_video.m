function [B_hat, Uo, Xhat3, Uo_track] = ...
    LRPR_prac_video(Params, Paramsrwf, Y, Afull, Afull_t, Afull_tk, Masks2)
Ysqrt = sqrt(Y);
%%contains some changes to consider the CDP setting. Check if it can be
%%made general to be able to handle simulated data too
%X_hat = zeros(100);
for  o = 1 :Params.tnew % Main loop
    fprintf('outer loop iteration %d\n', o)
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
        
        [Qu, ~] = BlockIter(Uhat, 50, Params.r); 
        Uo = Qu(:, 1 : Params.r);
        Uo_track{o} = Uo;
    end
    
    AUnew   =   zeros(Params.n_1*Params.n_2*Params.L,Params.q,Params.r);
    Ybk     =   zeros(Params.n_1,Params.n_2,Params.r,Params.q);
    
    for nr  =   1 : Params.r
        AUnew(:,:,nr) =  reshape(Afull(repmat(Uupdt(:,:,nr),[1,1,Params.q])), Params.n_1*Params.n_2*Params.L,Params.q);
        Ybk(:,:,nr,:) =  Afull_tk(Y.* Afull(repmat(Uupdt(:,:,nr), [1,1,Params.q])));
    end
    
    B_hat  =   zeros(Params.r, Params.q);
    
    %  Using Simple PR for estimating coefficients
    for ni = 1 : Params.q
        
        Masks  =   Masks2(:,:,:,ni);
        A_pr  = @(I)  reshape(fft2(conj(Masks) .* ...
            reshape(repmat(Uhat_vec * I, Params.L, 1), ...
            Params.n_1, Params.n_2, Params.L)), [], 1);
        At_pr = @(W) Params.n_1 * Params.n_2 * Uhat_vec' * reshape(sum(Masks .* ...
            ifft2(reshape(W, Params.n_1, Params.n_2, Params.L)), 3), [], 1);

        y_tmp = sqrt(reshape(Y(:, :, :, ni), [], 1));
        Paramsrwf.Tb_LRPRnew = Params.Tb_LRPRnew(o);
        [bhat] = RWFsimple(y_tmp, Paramsrwf, A_pr, At_pr);
        %size(bhat)
        B_hat(:, ni) = bhat;
        x_k =  Uo *  B_hat(:,ni);
%        Chat(:,ni) = (At_pr(repmat(x_k, Params.L, 1)) >= 0) ...
            %- (At_pr(repmat(x_k, Params.L, 1)) < 0);
        %X_hat(:, ni) = x_k;
    end
%     [Qb,~]  =  qr(B_hat');
%     Bo   =   Qb(:,1:Params.r)';
[Qb,~] = BlockIter(B_hat', 50, Params.r);
Bo = Qb(:, 1:Params.r)';
Xhat1    =   Uo*B_hat;
Xhat3   =   reshape(Xhat1,Params.n_1,Params.n_2,Params.q);
Chat     =   exp(1i*angle(Afull(Xhat3))); % Initial phase

    
    
    K1       =   Chat .* Ysqrt;
    Zvec     =   reshape(K1,Params.n_1*Params.n_2*Params.L*Params.q,1);
    Uvec    =   cgls_new(@mult_H2, @mult_Ht2 , Zvec, 0,1e-6 ,3);
    U_hat    =   reshape(Uvec, Params.n_1*Params.n_2, Params.r);
    
    
    [Qu,~]  =  BlockIter(U_hat, 10, Params.r);
    Uo  =  Qu(:, 1:Params.r);
    Uo_track{o} = Uo;
end
    function i_out = mult_H2(i_in)
        I_mat    =   reshape(i_in,(Params.n_1*Params.n_2), Params.r);
        % i_out    =   zeros(Params.q*Params.m, 1);
        Xmat        =    I_mat * Bo;
        Xmat2   =   reshape(Xmat,Params.n_1,Params.n_2,Params.q);
        Iout        =     Afull(Xmat2);
        i_out       =     reshape(Iout,Params.q*Params.m, 1);
    end

%   Defining mult_Ht

    function w_out = mult_Ht2(w_in)
        w_out   =   zeros(Params.n_1*Params.n_2*Params.r, 1);
        TmpVec  =    permute(Afull_tk(reshape(w_in, Params.n_1,Params.n_2,Params.L,Params.q)), [1,2,4,3]);
        for nk = 1: Params.q
            w_out    =   w_out + kron(Bo(:,nk), reshape(TmpVec(:,:,nk), Params.n_1*Params.n_2, 1));
        end
    end
end