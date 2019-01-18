function [Altmintime,Bhat, Uhat, Xhat2] = alt_min_init(Y, Params, Afull, Afull_t, Afull_tk)

tic;
Ytrk    =   zeros(Params.n_1,Params.n_2,Params.L,Params.q);
for ni = 1 : Params.q
    Yk  =   reshape(Y(:,:,:,ni), Params.n_1*Params.n_2*Params.L, 1);
    %normest =   sum(Yk(:))/numel(Yk);
    normest =   sum(Yk(:))/Params.m;
    Eyk     =   (Yk <= Params.alpha_y^2*normest);
    % num_larg = sum(Eyk);
    Ytrk(:,:,:,ni)  =   reshape(Eyk.*Yk,Params.n_1,Params.n_2,Params.L);
end

%/////////////////////////////////////////////////////////////////
% Block power method for estimating initial U

U_tmp1  = randn(Params.n_1*Params.n_2,Params.r);
%[Uupdt, v, w]   =   qr(reshape(randn(n_1, n_2, r), n_1*n_2, r), 0);
[U_upd_vec, ~, ~]   =   qr(U_tmp1, 0);

Uupdt          =   reshape(U_upd_vec, Params.n_1, Params.n_2, Params.r);

U_tmp   =  zeros(Params.n_1,Params.n_2,Params.r) ;

for t = 1 : Params.itr_num_pow_mth
    for nr  =   1 : Params.r
        U_tmp(:,:,nr)           = Afull_t( Ytrk.* Afull(repmat(Uupdt(:,:,nr), [1,1,Params.q])));
    end
    [Uupdt3, ~, ~]   =   qr(reshape(U_tmp, Params.n_1*Params.n_2, Params.r), 0);
    Uupdt          =   reshape(Uupdt3, Params.n_1, Params.n_2, Params.r);
    
    % Uupdt          =   reshape( U_tmp, Params.n_1, Params.n_2, Params.r);
end


Uhat_vec    =   reshape(Uupdt, Params.n_1*Params.n_2, Params.r);
Uhat            =    Uhat_vec;
%Err_U       =   norm(Uvec - Uhat_vec * Uhat_vec' * Uvec);
Altmintime(1) = toc;
%////////////////////////////////////////////////////////////////
%  Finding the initial estimate of B
tic;
AUnew   =   zeros(Params.n_1*Params.n_2*Params.L,Params.q,Params.r);
Ybk     =   zeros(Params.n_1,Params.n_2,Params.r,Params.q);
for nr  =   1 : Params.r
    AUnew(:,:,nr) =  reshape(Afull(repmat(Uupdt(:,:,nr),[1,1,Params.q])), Params.n_1*Params.n_2*Params.L,Params.q);
    Ybk(:,:,nr,:) =  Afull_tk(Y.* Afull(repmat(Uupdt(:,:,nr), [1,1,Params.q])));
end
Bhat     =  zeros(Params.r,Params.q);
for na = 1 : Params.q
    ybk = Uhat_vec' * reshape(Ybk(:,:,:,na),Params.n_1*Params.n_2,Params.r);
    [b_t1,~,~]   =   svd(ybk);
    nb =   sqrt(sum(reshape(Y(:,:,:,na),Params.n_1*Params.n_2*Params.L,1))/Params.m);
    nc = Params.m*Params.r/(norm(reshape(AUnew(:,na,:),Params.n_1*Params.n_2*Params.L,Params.r),'fro')^2);
    norm_b_a_hat1=   sqrt(nc)* nb;
    Bhat(:,na)   =   norm_b_a_hat1 * b_t1(:,1);
end
%///////////////////////////////////////////////////////////////
Altmintime(2)=toc;
%Chat    =   zeros(Params.m, Params.q);
tic;
Xhat    =   Uhat*Bhat;
Xhat2   =   reshape(Xhat,Params.n_1,Params.n_2,Params.q);



%Ysqrt   =   sqrt(Y);
%Ysqrt2  =   reshape(Ysqrt,Params.n_1*Params.n_2*Params.L,Params.q );
