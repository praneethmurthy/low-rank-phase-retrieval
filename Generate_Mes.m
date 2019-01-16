        
       function [Ysqrt,Y,A] = Generate_Mes(X,Params,m)
        %$$$$$
        % Generating design matrices and measurements
        %$$$$$
        A      =   zeros(Params.n, m, Params.q);% Design matrices
        Y      =   zeros(m, Params.q);% Matrix of measurements
        for nl = 1 : Params.q
            A(:, :, nl) = randn(Params.n, m);
            Y(:,nl)     = abs(A(:,:,nl)' * X(:,nl)).^2;
        end
        %$$$
        Ysqrt = sqrt(Y);