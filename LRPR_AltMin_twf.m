%%  Low Rank Phase Retrieval using Alternating Minimization
function [Xhat, Uhat, U_track, X_track, time_iter] = LRPR_AltMin_twf(Y, A, Params, Paramstwf)

time_iter = zeros(Params.told, 1);
Xhat    =   zeros(Params.n, Params.q);
ctr = 1;
%tic;
%%   Initialization
Yu  =   zeros(Params.n, Params.n);
normest = 9/(Params.m * Params.q) * sum(Y(:));
for k = 1 : Params.q
    %normest = sqrt((9/Params.m) * Y(:,k)' * Y(:, k));
    Ytr = Y(:,k) .* (abs(Y(:, k)) > normest);
    Yu 	=   Yu + A(:,:,k) * diag(Ytr) * A(:,:,k)';
end

Yu          =   Yu / Params.m / Params.q;
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
U_track{ctr} = Uhat;


Bhat        =   zeros(Params.r, Params.q);

for k = 1 : Params.q
    
    AU          =   A(:,:,k)' * Uhat ;
    Yb          =   AU' * diag(Y(:,k)) * AU;
    [V,~,~]     =   svd(Yb);
    Bhat(:,k)   =   sqrt(sum(Y(:,k))/Params.m) * V(:,1);
    Xhat(:,k)        =   Uhat * Bhat(:,k);
    
end
X_track{ctr} = Xhat;
ctr = ctr+1;
Chat    =   zeros(Params.m, Params.n);
for a = 1 : Params.q
    Chat(:,a)   =   exp(1i*angle(A(:,:,a)' * Uhat * Bhat(:,a)));
end

%time_iter(1) = toc;


%%need to invoke TWF here

Rel_Err = [];
%% Loop
grad_type = Paramstwf.grad_type;
if strcmp(grad_type, 'TWF_Poiss') == 1
    mu = @(t) Paramstwf.mu; % Schedule for step size
elseif strcmp(grad_type, 'WF_Poiss') == 1
    tau0 = 330;                         % Time constant for step size
    mu = @(t) min(1-exp(-t/tau0), 0.2); % Schedule for step size
end

for t = 1: Paramstwf.T
    for a = 1 : Paramstwf.alpha
        z = Xhat(:, a);
        z              =   Uhat * Uhat' * z;
        Aa      =   @(I) A(:,:,a) * I;
        Aa_t    =   @(Y) A(:,:,a)' * Y;
        grad(:,a) = compute_grad(z(:,a), Y(:,a), Paramstwf, Aa, Aa_t);
        z(:,a) = z(:,a) - mu(t) * grad(:,a);             % Gradient update
        % Relerrs2(a) = [norm(X(:,a) - exp(-1i*angle(trace(X(:,a)'*z(:,a)))) * z(:,a), 'fro')/norm(X(:,a),'fro')];
        % Xtwf(:,a,t) = z(:,a);
        
    end
    % Xtwf(:,:,t)        =   Xtwf ;
    [Utmp,~]    =   svd(z);
    Uhat                =   Utmp(:,1:Paramstwf.r);
    U_track{ctr} = Uhat;
    ctr = ctr+1;
    Xhat = Uhat * Uhat' * z;
    %fprintf('TWF Projection Error is:\t%2.2e\n', mean(Tmp_Err_X));
    for na = 1 : Paramstwf.alpha
        xa_hat      =   z(:,na);
        xa          =   X(:,na);
        %                 %  Tmp_Err_X(na)   =   min(norm(xa-xa_hat)^2, norm(xa+xa_hat)^2);
        Tmp_Err_X(na) 	=   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
    end
    Proj2_Err =  sum(Tmp_Err_X)/norm(X,'fro');
    if (Proj2_Err <= Treshold)
        return;
    end
end
%           Rel_Err(:,t)=  Tmp_Err_X;
%           fprintf('TWF Projection Error is:\t%2.2e\n', mean(Tmp_Err_X));

end
