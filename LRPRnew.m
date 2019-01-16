function [B_hat, U_hat] = LRPRnew(Params, Paramsrwf, Y, Ysqrt, A)

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
            U_hat  = P;
        end
        %%%%%
        
        B_hat  =   zeros(Params.r, Params.q);
        Chat   =   zeros(Params.m, Params.q);% Estimated phase
        %  Using Simple PR for estimating coefficients
        for ni = 1 : Params.q
            Amatrix  =  A(:,:,ni)' *  U_hat;% Design matrices for coefficients
            A1  = @(I) Amatrix  * I;
            At  = @(Y) Amatrix' * Y;
            %atrue = B(:,ni);
            %[a, Relerrs] = TWFsimple(Y(:,ni),atrue, Paramstwf, A1, At);
            [bhat] = RWFsimple(Ysqrt(:,ni), Paramsrwf, A1, At);
            B_hat(:,ni) = bhat;
            x_k =  U_hat *  B_hat(:,ni);
            Chat(:,ni) = (A(:,:,ni)'* x_k >= 0) - (A(:,:,ni)'* x_k < 0);
        end

        % Estimating the subspace
        SumS = zeros((Params.n*Params.r), (Params.n*Params.r));
        Sumg = zeros((Params.n*Params.r), 1);
        for nt = 1 : Params.q
            gt    =   Chat(:,nt).* sqrt(Y(:,nt));
            Mt    =   A(:,:,nt)' * kron(B_hat(:,nt)' , eye(Params.n));
            SumS  =   SumS + (Mt'*Mt);
            Sumg  =   Sumg + Mt' * gt;
        end
        Uhatn     =   SumS \ Sumg;
        U_hat   =   reshape(Uhatn, Params.n, Params.r);
    end
    



