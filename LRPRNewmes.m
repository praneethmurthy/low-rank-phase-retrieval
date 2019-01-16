function [B_hat, Uo, X_hat, Uo_track] = LRPRNewmes(Params, Paramsrwf, Y, Ysqrt, A, m_u, m_b, m_init, X)
[Ysqrt_init,~,Ai] = Generate_Mes(X,Params,m_init);
for  o = 1 :Params.tnew % Main loop
    %%%%%%%
    % Initializing the subspace
    %%%%%%%
    if o == 1
        Yu      =   zeros(Params.n, Params.n);
        for nh = 1 : Params.q
            Yu  =   Yu + Ai(:,:,nh) * diag(Ysqrt_init(:,nh)) * Ai(:,:,nh)';
        end
        Yu      =   Yu / Params.q / Params.m;
        [P,~,~] =   svds(Yu, Params.r);
        U_hat = P;
        Uo = U_hat;
    end
    Uo_track{o} = Uo;
    %%%%%
    [Ysqrt1,Y1,Ab] = Generate_Mes(X,Params,m_b);
    B_hat  =   zeros(Params.r, Params.q);
    [Ysqrt_u,Yu,Au] = Generate_Mes(X,Params,m_u);

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
    
    [Qb,Rb]  =  qr(B_hat');
    Bo   =   Qb(:,1:Params.r)';
    
    Zvec    =   zeros(Params.m*Params.q, 1);
    for nt = 1 : Params.q
        strt_idx    =   Params.m*(nt-1) + 1;
        end_idx     =   strt_idx + Params.m - 1;
        TempVec     =   Chat(:, nt) .* Ysqrt_u(:,nt);
        Zvec(strt_idx:end_idx, 1)   =   TempVec;
    end
    Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 1e-16,30);
    U_hat    =   reshape(Uvec, Params.n, Params.r);
    [Qu,Ru]  =  qr(U_hat);
    Uo  =  Qu(:, 1:Params.r);
    X_hat = Uo * B_hat;
    
end

    function x_out = mult_H(x_in)
        X_mat    =   reshape(x_in, Params.n, Params.r);
        %    x_out    =   A_long * X_vec;
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

%         %Estimating the subspace
%         SumS = zeros((Params.n*Params.r), (Params.n*Params.r));
%         Sumg = zeros((Params.n*Params.r), 1);
%         for nt = 1 : Params.q
%             x_k =  U_hat *  B_hat(:,nt);
%             X_hat(:, ni) = x_k;
%             Chat = (Au(:,:,nt)'* x_k >= 0) - (Au(:,:,nt)'* x_k < 0);
%             gt    =   Chat.* Ysqrt_u(:,nt);
%             Mt    =   Au(:,:,nt)' * kron(B_hat(:,nt)' , speye(Params.n));
%             SumS  =   SumS + (sparse(kron(B_hat(:,nt) , speye(Params.n))) * Au(:, :, nt) * ...
%                 Au(:,:,nt)' * sparse(kron(B_hat(:,nt)' , speye(Params.n))));
%             Sumg  =   Sumg + sparse(kron(B_hat(:,nt) , speye(Params.n))) * Au(:, :, nt)  * gt;
%         end
%         Uhatn     =   SumS \ Sumg;
%         U_hat   =   reshape(Uhatn, Params.n, Params.r);
end




