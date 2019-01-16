function [B_hat, Uo, X_hat] = LRPRNewmes(Params, Paramsrwf, Y, Ysqrt, A, m_u, m_b, m_init, X)

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
            U_hat = P;
        end
        %%%%%
        [Ysqrt1,Y1,Ab] = Generate_Mes(X,Params,m_b);
        B_hat  =   zeros(Params.r, Params.q);
        X_hat = zeros(Params.n, Params.q);
        %Chat   =   zeros(Params.m, Params.q);% Estimated phase
        %  Using Simple PR for estimating coefficients
        for ni = 1 : Params.q
            Amatrix  =  Ab(:,:,ni)' *  U_hat;% Design matrices for coefficients
            A1  = @(I) Amatrix  * I;
            At  = @(Y) Amatrix' * Y;
            %atrue = B(:,ni);
            %[a, Relerrs] = TWFsimple(Y(:,ni),atrue, Paramstwf, A1, At);
            [bhat] = RWFsimple(Ysqrt1(:,ni), Paramsrwf, A1, At);
            B_hat(:,ni) = bhat;
        end

         [Ysqrt_u,Yu,Au] = Generate_Mes(X,Params,m_u);
        % Estimating the subspace
        SumS = zeros((Params.n*Params.r), (Params.n*Params.r));
        Sumg = zeros((Params.n*Params.r), 1);
        for nt = 1 : Params.q
            x_k =  U_hat *  B_hat(:,nt);
            X_hat(:, ni) = x_k;
            Chat = (Au(:,:,nt)'* x_k >= 0) - (Au(:,:,nt)'* x_k < 0);
            gt    =   Chat.* Ysqrt_u(:,nt);
%             Mt    =   Au(:,:,nt)' * kron(B_hat(:,nt)' , speye(Params.n));
            SumS  =   SumS + (sparse(kron(B_hat(:,nt) , speye(Params.n))) * Au(:, :, nt) * ...
                Au(:,:,nt)' * sparse(kron(B_hat(:,nt)' , speye(Params.n))));
            Sumg  =   Sumg + sparse(kron(B_hat(:,nt) , speye(Params.n))) * Au(:, :, nt)  * gt;
        end
        Uhatn     =   SumS \ Sumg;
        U_hat   =   reshape(Uhatn, Params.n, Params.r);
        [Qu,Ru]  =  qr(U_hat);
        Uo  =  Qu(:, 1:Params.r);
    end
    



