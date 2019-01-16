function [B_hat, Uo, X_hat, Uo_track] = LRPRQR(Params, Paramsrwf, Y, Ysqrt, A)
 
for  o = 1 :Params.tnew % Main loop
    %%%%%%%
    % Initializing the subspace
    %%%%%%%
    if o == 1
        Yu      =   zeros(Params.n, Params.n);
        for nh = 1 : Params.q
            Yu  =   Yu + A(:,:,nh) * diag(Y(:,nh)) * A(:,:,nh)';
        end
        Yu      =   Yu / Params.q / Params.m;
        [P,~,~] =   svds(Yu, Params.r);
        Uo = P;
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
    [Qb,Rb]  =  qr(B_hat');
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
    Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 1e-16,30);
    U_hat    =   reshape(Uvec, Params.n, Params.r);
    
    
    
    %         SumS = zeros((Params.n*Params.r), (Params.n*Params.r));
    %         Sumg = zeros((Params.n*Params.r), 1);
    %         for nt = 1 : Params.q
    %             gt    =   Chat(:,nt).* sqrt(Y(:,nt));
    % %             Mt    =   A(:,:,nt)' * kron(Bo(:,nt)' , speye(Params.n));
    % %             size(Mt)
    % %             size(A(:,:,nt))
    % %             size(kron(Bo(:,nt)' , speye(Params.n)))
    %             SumS  =   SumS + (sparse(kron(Bo(:,nt) , speye(Params.n))) * A(:, :, nt) * ...
    %                 A(:,:,nt)' * sparse(kron(Bo(:,nt)' , speye(Params.n))));
    %             Sumg  =   Sumg + sparse(kron(Bo(:,nt) , speye(Params.n))) * A(:, :, nt)  * gt;
    %         end
    %         Uhatn     =   SumS \ Sumg;
    %         U_hat   =   reshape(Uhatn, Params.n, Params.r);
    [Qu,Ru]  =  qr(U_hat);
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

    



