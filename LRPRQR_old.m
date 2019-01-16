function [B_hat, Uo, X_hat] = LRPRQR(Params, Paramsrwf, Y, Ysqrt, A)
    
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
            [bhat] = RWFsimple(Ysqrt(:,ni), Paramsrwf, A1, At);
            B_hat(:,ni) = bhat;
            x_k =  Uo *  B_hat(:,ni);
            Chat(:,ni) = (A(:,:,ni)'* x_k >= 0) - (A(:,:,ni)'* x_k < 0);
            X_hat(:, ni) = x_k;
        end
        [Qb,Rb]  =  qr(B_hat');
        Bo   =   Qb(:,1:Params.r)';
        

        % Estimating the subspace
        SumS = zeros((Params.n*Params.r), (Params.n*Params.r));
        Sumg = zeros((Params.n*Params.r), 1);
        for nt = 1 : Params.q
            gt    =   Chat(:,nt).* sqrt(Y(:,nt));
%             Mt    =   A(:,:,nt)' * kron(Bo(:,nt)' , speye(Params.n));
%             size(Mt)
%             size(A(:,:,nt))
%             size(kron(Bo(:,nt)' , speye(Params.n)))
            SumS  =   SumS + (sparse(kron(Bo(:,nt) , speye(Params.n))) * A(:, :, nt) * ...
                A(:,:,nt)' * sparse(kron(Bo(:,nt)' , speye(Params.n))));
            Sumg  =   Sumg + sparse(kron(Bo(:,nt) , speye(Params.n))) * A(:, :, nt)  * gt;
        end
        Uhatn     =   SumS \ Sumg;
        U_hat   =   reshape(Uhatn, Params.n, Params.r);
        [Qu,Ru]  =  qr(U_hat);
        Uo  =  Qu(:, 1:Params.r);
    end
    



