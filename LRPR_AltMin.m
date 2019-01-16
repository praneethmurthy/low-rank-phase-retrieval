%%  Low Rank Phase Retrieval using Alternating Minimization
function [Xhat, Uhat, U_track] = LRPR_AltMin(Y, A, Params)


% ExeTime     =   zeros(Params.T1+1, 1);     %   exe time
% ErX         =   zeros(Params.T1+1, 1);
% ErU         =   zeros(Params.T1+1, 1);

% Xopt    =   Uopt * Bopt;
% nrm_X   =   norm(Xopt, 'fro')^2;
Xhat    =   zeros(Params.n, Params.q);

%%   Initialization
tic;
Yu  =   zeros(Params.n, Params.n);
for k = 1 : Params.q
    Yu 	=   Yu + A(:,:,k) * diag(Y(:,k)) * A(:,:,k)';
end

Yu          =   Yu / Params.m / Params.q;
[Ui, ~, ~]  =   svd(Yu);
Uhat        =   Ui(:, 1:Params.r);
% ErU(1)      =   abs(sin(subspace(Uopt,Uhat)));

% TmpEr       =   0;
Bhat        =   zeros(Params.r, Params.q);

for k = 1 : Params.q
  
    AU          =   A(:,:,k)' * Uhat ;
    Yb          =   AU' * diag(Y(:,k)) * AU;
    [V,~,~]     =   svd(Yb);
    Bhat(:,k)   =   sqrt(sum(Y(:,k))/Params.m) * V(:,1);
    Xhat(:,k)        =   Uhat * Bhat(:,k);
%     x           =   Uopt * Bopt(:,k);
%     TmpEr       =   TmpEr  + min(norm(x - xhat)^2, norm(x + xhat)^2);

end
% ErX(1)      =   TmpEr / nrm_X;
Chat    =   zeros(Params.m, Params.n);
for a = 1 : Params.q
    Chat(:,a)   =   exp(1i*angle(A(:,:,a)' * Uhat * Bhat(:,a)));
end

% ExeTime(1)  =   toc;

%%  Main Loop

Ysqrt   =   sqrt(Y);
Zvec    =   zeros(Params.m*Params.q, 1);
for t = 1 : Params.told
    U_track{t} = Uhat;
    %tic;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       Updating Uhat
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for a = 1 : Params.q
        
        strt_idx    =   Params.m*(a-1) + 1;
        end_idx     =   strt_idx + Params.m - 1;
        TempVec     =   Chat(:,a) .* Ysqrt(:,a);
        Zvec(strt_idx:end_idx, 1)   =   TempVec;
        
    end
%     Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 10^-14,20);
    Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 10^-16,40);
    Uhat    =   reshape(Uvec, Params.n, Params.r);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       Updating Bhat and Chat
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     ErrTmp  =   0;
    for  a = 1 : Params.q
        
        R           =   A(:,:,a)' * Uhat;
        K           =   Chat(:,a) .* Ysqrt(:,a);
        %         Bhat(:,a)   =   R \ K ;
        Bhat(:,a)   =   mldivide(R , K );
        Chat(:,a)   =   exp(1i*angle(R * Bhat(:,a)));
        Xhat(:,a)      =   Uhat * Bhat(:, a);
%         ErrTmp      =   ErrTmp + min(norm(xhat_k-Xopt(:,a))^2, norm(xhat_k+Xopt(:,a))^2);
        
    end
    
    %Xhat(:,:,t) = Uhat * Bhat;
%     ErX(t+1)    =   ErrTmp / nrm_X;
%     ErU(t+1)    =   abs(sin(subspace(Uopt,Uhat)));
%     ExeTime(t+1)=   toc;
    
end

%Xhat    =   Uhat * Bhat;


    function x_out = mult_H(x_in)
        X_mat    =   reshape(x_in, Params.n, Params.r);
        %    x_out    =   A_long * X_vec;
        x_out    =   zeros(Params.q*Params.m, 1);
        for na = 1: Params.q
            x_out((na-1)*Params.m + 1 : na*Params.m) = A(:,:,na)' * X_mat * Bhat(:,na);
        end
        
    end


    function w_out = mult_Ht(w_in)
        
        w_out   =   zeros(Params.n*Params.r, 1);
        for na = 1: Params.q
            tmp_vec  =   A(:,:,na) * w_in((na-1)*Params.m+1:na*Params.m);
            w_out    =   w_out + kron(Bhat(:,na), tmp_vec);
        end
        
    end

end
