%%  Low Rank Phase Retrieval using Alternating Minimization
function [Xhat, Uhat, U_track, X_track, time_iter] = LRPR_AltMin1(Y, Ysqrt, A, Params)

time_iter = zeros(Params.told, 1);
Xhat    =   zeros(Params.n, Params.q);

%tic;
%%   Initialization
Yu  =   zeros(Params.n, Params.n);
normest = 9/(mean(Params.m) * Params.q) * sum(cell2mat(Y));
        for nh = 1 : Params.q
            %normest = sqrt((13/Params.m_init) * Y(:,nh)' * Y(:, nh));
            Ytr = Y{nh} .* (abs(Y{nh}) > normest);
            Yu  =   Yu + A{nh} * diag(Ytr) * A{nh}';
        end
        Yu      =   Yu / Params.q / mean(Params.m);

if(Params.rank_est_flag == 1) %%need to estimate rank
    %%checking rank estimation
    [~,sig_init,~] =   svd(Yu);
    sig_init = diag(sig_init);
    tmp1 = 1.* (sig_init(1:end-1) - min(sig_init) >= 1.3 * min(Params.sig_star)^2/Params.q);
    if (all(tmp1 == 0))
        est_rank = 1;
    else
        est_rank = find(tmp1 == 1, 1, 'last');
    end
    Params.r = est_rank;
    fprintf('estimated rank is %d\n', Params.r);
end
[Ui, ~, ~]  =   svd(Yu);
Uhat        =   Ui(:, 1:Params.r);
Bhat        =   zeros(Params.r, Params.q);

for k = 1 : Params.q
    
    AU          =   A{k}' * Uhat ;
    Yb          =   AU' * diag(Y{k}) * AU;
    [V,~,~]     =   svd(Yb);
    Bhat(:,k)   =   sqrt(sum(Y{k})/Params.m(k)) * V(:,1);
    Xhat(:,k)        =   Uhat * Bhat(:,k);
    
end
% Chat    =   zeros(Params.m, Params.n);
Chat    =   cell(Params.q, 1);
for a = 1 : Params.q
    Chat{a}   =   exp(1i*angle(A{a}' * Uhat * Bhat(:,a)));
end

%time_iter(1) = toc;

%%  Main Loop
%Ysqrt   =   sqrt(Y);
Zvec    =   zeros(mean(Params.m)*Params.q, 1);
tic;
for t = 1 : Params.told
    U_track{t} = Uhat;
    X_track{t} = Uhat * Bhat;
    %tic;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       Updating Uhat
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for a = 1 : Params.q
        
        strt_idx    =   Params.m(a)*(a-1) + 1;
        end_idx     =   strt_idx + Params.m(a) - 1;
        TempVec     =   Chat{a} .* Ysqrt{a};
        Zvec(strt_idx:end_idx, 1)   =   TempVec;
        
    end
    %     Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 10^-14,20);
    Uvec    =   cgls_new(@mult_H, @mult_Ht , Zvec, 0, 10^-16,30);
    Uhat    =   reshape(Uvec, Params.n, Params.r);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       Updating Bhat and Chat
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for  a = 1 : Params.q
        R           =   A{a}' * Uhat;
        K           =   Chat{a} .* Ysqrt{a};
        Bhat(:,a)   =   mldivide(R , K );
        Chat{a}   =   exp(1i*angle(R * Bhat(:,a)));
        Xhat(:,a)   =   Uhat * Bhat(:, a);
        
    end
    time_iter(t) = toc;
end

%%%% helper functions
    function x_out = mult_H(x_in)
        X_mat    =   reshape(x_in, Params.n, Params.r);
        %    x_out    =   A_long * X_vec;
        x_out    =   zeros(Params.q*mean(Params.m), 1);
        for na = 1: Params.q
            x_out((na-1)*Params.m(na) + 1 : na*Params.m(na)) = A{na}' * X_mat * Bhat(:,na);
        end
    end


    function w_out = mult_Ht(w_in)
        w_out   =   zeros(Params.n*Params.r, 1);
        for na = 1: Params.q
            tmp_vec  =   A{na}* w_in((na-1)*Params.m(na)+1:na*Params.m(na));
            w_out    =   w_out + kron(Bhat(:,na), tmp_vec);
        end
    end
end