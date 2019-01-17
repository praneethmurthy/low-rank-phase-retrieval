%%checking heuristic for subspace change detection
close all
clear;
clc;

const_thresh_minus = zeros(1, 200);
const_thresh_plus = zeros(1, 200);
for mc = 1 : 200
    fprintf('==== Iteration %d\n', mc);
    Params.Tmont = 30;
    
    Params.n  =  200;   % Number of rows of the low rank matrix
    Params.q  =  200;   % Number of columns of the matrix for LRPR
    Params.r  =  2;     % Rank
    Params.m = 85;     % Number of measurements
    
    Params.tnew = 10;    % Total number of main loops of new LRPR
    Params.told = 10;    % Total number of main loops of Old LRPR
    
    m_b = Params.m;          %Number of measuremnets for coefficient estimate
    m_u = Params.m;           % Number of measuremnets for subspace estimate
    m_init = Params.m;       % Number of measuremnets for init of subspace
    %m_init = 50;
    
    %Params.m  =  m_init + (m_b+m_u)*Params.tot;% Number of measurements
    
    %%~PN editing m, n, r so that the variables are globally same
    % TWF Parameters
    Paramsrwf.m  =  Params.m;% Number of measurements
    Paramsrwf.n  =  Params.n;% size of columns of coefficient matrix or x_k
    Paramsrwf.r  =  Params.r;% size of columns of coefficient matrix or b_k
    Paramsrwf.npower_iter = 50;% Number of loops for initialization of TWF with power method
    Paramsrwf.mu          = 0.2;% Parameter for gradient
    %Params.Tb_LRPRnew    = unique(ceil(linspace(30, 100, Params.tnew)));% Number of loops for b_k with simple PR
    Params.Tb_LRPRnew    = 85 * ones(1, Params.tnew);
    % Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
    Paramsrwf.TRWF           = 25;% Number of loops for b_k with simple PR
    Paramsrwf.cplx_flag   = 0;
    % Paramstwf.alpha_y     = 3;
    % Paramstwf.alpha_h     = 5;
    % Paramstwf.alpha_ub    = 5;
    % Paramstwf.alpha_lb    = 0.3;
    % Paramstwf.grad_type   = 'TWF_Poiss';
    %Params.seed = rng;
    err_SE_iter = zeros(3, Params.tnew, Params.Tmont);
    
    %generate true data
    U0 = orth(randn(Params.n, Params.r));
    B = randn(Params.n);
    B1 = (B - B')/2;
    U1 = expm(0.001 * B1) * U0;
    
    se_true=abs(sin(subspace(U0, U1)));
    
    Btrue_0 = randn(Params.r, 2 * Params.q);
    X = [U0 * Btrue_0(:, 1 : Params.q), U1 * Btrue_0(:, Params.q + 1 : end)];
    
    %generate measurements
    A      =   zeros(Params.n, Params.m, 2 * Params.q);% Design matrices
    Y      =   zeros(Params.m, Params.q);% Matrix of measurements
    for nl = 1 : 2 * Params.q
        A(:, :, nl) = randn(Params.n, Params.m);
        Y(:,nl)     = abs(A(:,:,nl)' * X(:,nl)).^2;
    end
    Ysqrt = sqrt(Y);
    
    Yu = zeros(Params.n);
    for nh = Params.q + 1 : 2 * Params.q
        Yu  =   Yu + A(:,:,nh) * diag(Y(:,nh)) * A(:,:,nh)';
    end
    Yu      =   Yu / Params.q / Params.m;
    
    U0_perp = eye(Params.n) - U0 * U0';
    [u2, s2] = svds(X);
    smax_true = s2(1,1);
    
    [u1,s1] = svds(U0_perp * Yu);
    
    svals = diag(s1);
    const_thresh_plus(mc) = (svals(1) + svals(end))* Params.q/smax_true;
    const_thresh_minus(mc) = (svals(1) - svals(end))* Params.q/smax_true;
end


figure
subplot(211)
hist(const_thresh_plus, 10)
title('c plus')
subplot(212)
hist(const_thresh_minus, 10)
title('c minus')