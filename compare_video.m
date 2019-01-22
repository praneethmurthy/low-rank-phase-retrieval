clc;
clear;
close all;

ob = VideoReader('Mouse.mp4');
%ob = VideoReader('videos/sara.mp4');
vidFrames = read(ob);
%numFrames = get(ob, 'numberOfFrames');
numFrames  =size(vidFrames,4);

for i = 1: numFrames
    temp = vidFrames(:,:,:,i);
    temp = rgb2gray(temp);
    temp = double(temp);
    temp = imresize(temp,0.15);
    p=size(temp,1);
    d=size(temp,2);
    I(:,i)=reshape(temp,[p*d,1]);
end

n_1     =   p;
n_2     =   d;
r       =   25;
q       =  numFrames ;
MaxIter  =  50;
X      =    I(:, 1 : q);
L       =   5;
numFrames = q;

alpha_y =   3;

itr_num_pow_mth    =  50;

Params.itr_num_pow_mth = itr_num_pow_mth;
Params.n_1 = n_1;
Params.n_2 = n_2;
Paramsrwf.n_1 = n_1;
Paramsrwf.n_2 = n_2;
Params.L = L;
Params.alpha_y = alpha_y;

Params.n  =  n_1 * n_2;   % Number of rows of the low rank matrix
Params.q  =  q;   % Number of columns of the matrix for LRPR
Params.r  =  25;     % Rank
Params.m       =   n_1*n_2*L;     % Number of measurements

Params.tnew = 30;    % Total number of main loops of new LRPR
Params.told = 5;    % Total number of main loops of Old LRPR

Params.m_b = Params.m;          %Number of measuremnets for coefficient estimate
Params.m_u = Params.m;           % Number of measuremnets for subspace estimate
Params.m_init = Params.m;       % Number of measuremnets for init of subspace
%m_init = 50;

% TWF Parameters
Paramsrwf.m  =  Params.m;% Number of measurements
Paramsrwf.n  =  Params.n;% size of columns of coefficient matrix or x_k
Paramsrwf.r  =  Params.r;% size of columns of coefficient matrix or b_k
Paramsrwf.npower_iter = 100;% Number of loops for initialization of TWF with power method
Paramsrwf.mu          = 0.2;% Parameter for gradient
Params.Tb_LRPRnew    = unique(ceil(linspace(7, 40, Params.tnew)));% Number of loops for b_k with simple PR
%Params.Tb_LRPRnew    = 60 * ones(1, Params.tnew);
% Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
Paramsrwf.TRWF           = 300;% Number of loops for b_k with simple PR
Paramsrwf.cplx_flag   = 1;


Den_X      =   norm(X,'fro');
frm_count  =   0;
XX         =   reshape(X,n_1,n_2,numFrames);
Xmat = XX;
TmpMask    =    zeros(n_1*n_2,q,L);

for nl = 1 : L
    TmpMask(:,:,nl) 	=   randsrc(n_1*n_2, q, [1,-1,1i,-1i]);
    %TmpMask(:,:,nl) 	=   randn(n_1*n_2, q);
end

%  Defining functions
Masks   =   reshape(TmpMask, n_1,n_2,L,q);

Afull 	 =	@(I) fft2(Masks.*reshape(repmat(I,[1,L]), n_1, n_2, L, q));
Afull_t  =	@(E) sum(sum( conj(Masks) .* ifft2(E), 3) , 4)* n_1 * n_2; %* size(E,3);
Afull_tk =	@(S) sum( conj(Masks) .* ifft2(S), 3)*n_1*n_2;% * size(E,1) * size(E,2) * size(E,3);


A_U = @(I, Uo) reshape(Afull(reshape(Uo * I, Params.n_1, Params.n_2, Params.q)), [], Params.q);
At_U = @(W, Uo) Uo' * ...
    reshape(Afull_tk(reshape(W, Params.n_1, Params.n_2, Params.L, Params.q)), [], Params.q) ;

% Afull 	 =	@(I) (Masks.*reshape(repmat(I,[1,L]), n_1, n_2, L, q));
% Afull_t  =	@(E) sum(sum( conj(Masks) .* (E), 3) , 4)* n_1 * n_2; %* size(E,3);
% Afull_tk =	@(S) sum( conj(Masks) .* (S), 3)*n_1*n_2;% * size(E,1) * size(E,2) * size(E,3);

%/////////////////////////////////////////////////
% Generating measurements

Y       =   abs(Afull(Xmat)).^2;

fprintf('data generation complete\n');

% Masks1    =  Masks2;
% Afull 	 =	@(I) fft2(conj(Masks1).*reshape(repmat(I,[1,L]), n_1, n_2, L, q));
% Afull_t  =	@(E) sum(sum( Masks1 .* ifft2(E), 3) , 4)* n_1 * n_2; %* size(E,3);
% Afull_tk =	@(S) sum( Masks1 .* ifft2(S), 3)*n_1*n_2;% * size(E,1) * size(E,2) * size(E,3);

% [B_hat, U_hat, Xhat, Uo_track, err_new, time_new] ...
%     = LRPR_prac_video_new(Params, Paramsrwf, Y, Afull, Afull_t, Afull_tk, X);
% [Altmintime,Bhat, Uhat,Xhat] = alt_min_init(Y, Params, Afull, Afull_t, Afull_tk);

% [B_hat, U_hat, Xhat, Uo_track, err_mc] ...
%     = LRPR_video_model_corr(Params, Paramsrwf, Y, Afull, Afull_t, Afull_tk, Masks, X);

Xhat_rwf = zeros(Params.n_1 * Params.n_2, Params.q);
err_rwf = zeros(Paramsrwf.TRWF+1, Params.q);
time_rwf = zeros(Paramsrwf.TRWF+1, Params.q);
for ni = 1 : Params.q
    Masks2  =   Masks(:,:,:,ni);
    ytmp = sqrt(reshape(Y(:, :, :, ni), [], 1));
    A_pr  = @(I)  reshape(fft2(Masks2 .* ...
        reshape(repmat(I, Params.L, 1), Params.n_1, Params.n_2, Params.L)), [],1);
    At_pr = @(W) 1 / (Params.n_1 * Params.n_2) * reshape(sum(conj(Masks2) .* ...
        ifft2(reshape(W, Params.n_1, Params.n_2, Params.L)), 3), [], 1);
    Paramsrwf.Tb_LRPRnew = 100;
    Paramsrwf.r = Params.n_1 * Params.n_2;
    [what_mc, errtmp, timetmp] = RWFsimple2_vid(ytmp, Paramsrwf, A_pr, At_pr, X(:, ni));
    err_rwf(:, ni) = errtmp;
    time_rwf(:, ni) = timetmp;
    Xhat_rwf(:, ni) = what_mc;
    %x_k =  Uo *  B_hat(:,ni);
    %Chat3 = exp(1i * angle(A_pr(Xhat_MC(:,ni))));
    %Xhat3(:, ni) = x_k;
    %Chat(:, :, :, ni) = reshape(Chat3, Params.n_1, Params.n_2, Params.L, 1);
end

Den_X      =   norm(X,'fro');
Tmp_Err_X2   =   zeros(Params.q, 1);
for   ct    =  1  :   Params.q
    xa_hat        =   Xhat(:,ct);
    xa            =   X(:,ct);
    Tmp_Err_X2(ct)  =   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
end
Nom_Err_X_twf	    =   sum(Tmp_Err_X2);

err_fin_rwf = mean(err_rwf, 2);
time_fin_rwf = mean(time_rwf, 2);

% figure
% loglog(time_fin_rwf, err_fin_rwf);

% figure
% loglog(time_new, err_new);


vdo_out_obj =   VideoWriter('m5n_rwf_plane');
open(vdo_out_obj);
Tmp_Err_X2   =   zeros(q, 1);
for   t    =  1  :   q
    tmpframe = reshape(Xhat_rwf(:, t), Params.n_1, Params.n_2);
    writeVideo(vdo_out_obj, uint8(abs(tmpframe)));
%     xa_hat        =   DD(:,t);
%     xa            =   X(:,t);
%     Tmp_Err_X2(t)  =   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
end
%Nom_Err_X_twf	    =   sum(Tmp_Err_X2);
%ERRTWFP             =  Nom_Err_X_twf / Den_X;
close(vdo_out_obj);

save('data/plane_rwf_5n_err_time.mat', 'Xhat_rwf', 'err_fin_rwf', 'time_fin_rwf')