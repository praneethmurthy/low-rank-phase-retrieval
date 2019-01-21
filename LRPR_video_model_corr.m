function [B_hat, Uo, Xhat_MC, Uo_track] = ...
    LRPR_video_model_corr(Params, Paramsrwf, Y, Afull, Afull_t, Afull_tk, Masks2, X)
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
        fprintf('initialization\n');
        %Den_X      =   norm(X,'fro');
        
        %truncating the measurements
        Ytrk    =   zeros(Params.n_1,Params.n_2,Params.L,Params.q);
        for ni = 1 : Params.q
            Yk  =   reshape( Y(:,:,:,ni) , Params.n_1*Params.n_2*Params.L, 1);
            normest =   sum(Yk(:))/Params.m;
            Eyk     =   ( Yk <= Params.alpha_y^2 *normest);
            Ytrk(:,:,:,ni)  =   reshape(Eyk.*Yk,Params.n_1,Params.n_2,Params.L);
        end
        
        %initializing U
        U_tmp1  = randn(Params.n_1*Params.n_2,Params.r);
        [U_upd_vec, ~, ~]   =   qr(U_tmp1, 0);
        Uupdt          =   reshape(U_upd_vec, Params.n_1, Params.n_2, Params.r);
        U_tmp   =  zeros(Params.n_1,Params.n_2,Params.r);
        
        for t = 1 : Params.itr_num_pow_mth
            %fprintf('power method iteration %d\n', t);
            for nr  =   1 : Params.r
                U_tmp(:,:,nr) = Afull_t( Ytrk.* Afull(repmat(Uupdt(:,:,nr), [1,1,Params.q])));
            end
            [Uupdt3, ~, ~]   =   qr(reshape(U_tmp, Params.n_1*Params.n_2, Params.r), 0);
            Uupdt            =     reshape(Uupdt3, Params.n_1, Params.n_2, Params.r);
        end
        Uhat_vec    =   reshape(Uupdt, Params.n_1*Params.n_2, Params.r);
        Uhat            =    Uhat_vec;
        %[Qu, ~] = BlockIter(Uhat, 100, Params.r);
        [Qu, ~] = qr(Uhat, 0);
        Uo = Qu(:, 1 : Params.r);
        Uo_track{o} = Uo;
    end
    
    
    %     AUnew   =   zeros(Params.n_1*Params.n_2*Params.L,Params.q,Params.r);
    %     Ybk     =   zeros(Params.n_1,Params.n_2,Params.r,Params.q);
    %
    %     for nr  =   1 : Params.r
    %         AUnew(:,:,nr) =  reshape(Afull(repmat(Uupdt(:,:,nr),[1,1,Params.q])), Params.n_1*Params.n_2*Params.L,Params.q);
    %         Ybk(:,:,nr,:) =  Afull_tk(Y.* Afull(repmat(Uupdt(:,:,nr), [1,1,Params.q])));
    %     end
    
    B_hat  =   zeros(Params.r, Params.q);
    %Chat = zeros(Params.n_1 * Params.n_2, Params.q);
    %Xhat3 = zeros(Params.n_1 * Params.n_2, Params.q);
    Chat = zeros(size(Y));
    for ni = 1 : Params.q
        %fprintf('RWF for %d\n', ni);
        Masks  =   Masks2(:,:,:,ni);
        A_pr  = @(I)  reshape(fft2( Masks .* ...
            reshape(repmat(Uo, Params.L, 1) * I, Params.n_1, Params.n_2, Params.L)), [],1);
        At_pr = @(W) Params.n_1 * Params.n_2 * Uo' * reshape(sum(conj(Masks) .* ...
            ifft2(reshape(W, Params.n_1, Params.n_2, Params.L)), 3), [], 1);
        
        y_tmp = reshape(Y(:, :, :, ni), [], 1);
        Paramsrwf.Tb_LRPRnew = Params.Tb_LRPRnew(o);
        [bhat] = RWFsimple(sqrt(y_tmp), Paramsrwf, A_pr, At_pr);
        B_hat(:, ni) = bhat;
        %x_k =  Uo *  B_hat(:,ni);
        Chat3 = exp(1i * angle(A_pr(B_hat(:,ni))));
        %Xhat3(:, ni) = x_k;
        Chat(:, :, :, ni) = reshape(Chat3, Params.n_1, Params.n_2, Params.L, 1); %reshape(repmat(Chat3, Params.L, 1), Params.n_1, Params.n_2, Params.L, 1);
    end
    
    Xhat3 = Uo * B_hat;
    Den_X      =   norm(X,'fro');
    Tmp_Err_X2   =   zeros(Params.q, 1);
    for   ct    =  1  :   Params.q
             xa_hat        =   Xhat3(:,ct);
             xa            =   X(:,ct);
            Tmp_Err_X2(ct)  =   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
    end
    Nom_Err_X_twf	    =   sum(Tmp_Err_X2);
    err_iter(o)             =  Nom_Err_X_twf / Den_X
    
    %     [Qb,~] = BlockIter(B_hat', 100, Params.r);
    [Qb,~] = qr(B_hat', 0);% 100, Params.r);
    Bo = Qb(:, 1:Params.r)';
    
    if (o==1)
        D      =    Uo*B_hat;
        Den_X      =   norm(X,'fro');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%initialization error
        for na = 1 : Params.q
            xa_hat      =   D(:,na);
            xa          =   X(:,na);
            %                 %  Tmp_Err_X(na)   =   min(norm(xa-xa_hat)^2, norm(xa+xa_hat)^2);
            Tmp_Err_X1(na) 	=   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
        end
        % Rel_Err(:,t)=  Tmp_Err_X;
        Err      = sum(Tmp_Err_X1);
        ERRinit     =   Err / Den_X;
        fprintf('Our initialization Error is:\t%2.2e\n',ERRinit);
    end
    
    
    K1       =   Chat .* Ysqrt; %sqrt;
    Zvec     =   reshape(K1,Params.n_1*Params.n_2*Params.L*Params.q,1);
    Uvec    =   cgls_new(@mult_H2, @mult_Ht2 , Zvec, 0,1e-6 ,3);
    U_hat    =   reshape(Uvec, Params.n_1*Params.n_2, Params.r);
    
    
    %[Qu,~]  =  BlockIter(U_hat, 100, Params.r);
    [Qu,~] = qr(U_hat, 0);
    Uo  =  Qu(:, 1:Params.r);
    Uo_track{o} = Uo;
end

%%%model correction step to improve residual error
fprintf('model correction step\n')
Xhat_MC = zeros(Params.n_1 * Params.n_2, Params.q);
for ni = 1 : Params.q
    Masks  =   Masks2(:,:,:,ni);
    xk = Xhat3(:, ni);
    ytmp = sqrt(reshape(Y(:, :, :, ni), [], 1));
    A_pr  = @(I)  reshape(fft2(Masks .* ...
        reshape(repmat(I, Params.L, 1), Params.n_1, Params.n_2, Params.L)), [],1);
    At_pr = @(W) 1 / (Params.n_1 * Params.n_2) * reshape(sum(conj(Masks) .* ...
        ifft2(reshape(W, Params.n_1, Params.n_2, Params.L)), 3), [], 1);
    Paramsrwf.Tb_LRPRnew = 30;
    Paramsrwf.r = Params.n_1 * Params.n_2;
    [what_mc] = RWFsimple(ytmp - A_pr(xk), Paramsrwf, A_pr, At_pr);
    Xhat_MC(:, ni) = - what_mc + xk;
    %x_k =  Uo *  B_hat(:,ni);
    %Chat3 = exp(1i * angle(A_pr(Xhat_MC(:,ni))));
    %Xhat3(:, ni) = x_k;
    %Chat(:, :, :, ni) = reshape(Chat3, Params.n_1, Params.n_2, Params.L, 1);
end

Den_X      =   norm(X,'fro');
Tmp_Err_X2   =   zeros(Params.q, 1);
for   ct    =  1  :   Params.q
    xa_hat        =   Xhat_MC(:,ct);
    xa            =   X(:,ct);
    Tmp_Err_X2(ct)  =   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
end
Nom_Err_X_twf	    =   sum(Tmp_Err_X2);
err_iter            =  Nom_Err_X_twf / Den_X

    function i_out = mult_H2(i_in)
        I_mat    =   reshape(i_in, Params.n_1*Params.n_2, Params.r);
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