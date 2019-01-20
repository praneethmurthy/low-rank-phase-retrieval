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
L       =   3;
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
%Params.Tb_LRPRnew    = unique(ceil(linspace(15, 25, Params.tnew)));% Number of loops for b_k with simple PR
Params.Tb_LRPRnew    = 60 * ones(1, Params.tnew);
% Paramsrwf.Tb_LRPRnew    = 85;% Number of loops for b_k with simple PR
Paramsrwf.TRWF           = 85;% Number of loops for b_k with simple PR
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

[B_hat, U_hat, Xhat, Uo_track] ...
    = LRPR_prac_video_new(Params, Paramsrwf, Y, Afull, Afull_t, Afull_tk, Masks, X);
%[Altmintime,Bhat, Uhat,Xhat] = alt_min_init(Y, Params, Afull, Afull_t, Afull_tk);
vdo_out_obj =   VideoWriter('out_30iter_rwf60_bef');
open(vdo_out_obj);
Tmp_Err_X2   =   zeros(q, 1);
for   t    =  1  :   q
    tmpframe = reshape(Xhat(:, t), Params.n_1, Params.n_2);
    writeVideo(vdo_out_obj, uint8(abs(tmpframe)));
%     xa_hat        =   DD(:,t);
%     xa            =   X(:,t);
%     Tmp_Err_X2(t)  =   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
end
%Nom_Err_X_twf	    =   sum(Tmp_Err_X2);
%ERRTWFP             =  Nom_Err_X_twf / Den_X;
close(vdo_out_obj);

% D       =        reshape(Xhat,n,q);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for na = 1 : Params.q
%     xa_hat      =   D(:,na);
%     xa          =   X(:,na);
%     %                 %  Tmp_Err_X(na)   =   min(norm(xa-xa_hat)^2, norm(xa+xa_hat)^2);
%     Tmp_Err_X1(na) 	=   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
% end
% % Rel_Err(:,t)=  Tmp_Err_X;
% % Err      = sum(Tmp_Err_X1);
% % ERRinit     =   Err / Den_X;
% % fprintf('Our initialization Error is:\t%2.2e\n',ERRinit);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% % Z      =    Xhat;
% % tic;
% % for t = 1: Paramstwf.T
% %     fprintf('Iteration # %d of main loop...\n', t);
% %     for   aa   =   1  :  q
% %         Masks  =   Masks2(:,:,:,aa);
% %         A  = @(I)  fft2(conj(Masks) .* reshape(repmat(I,[1 Paramstwf.L]), size(I,1), size(I,2), Paramstwf.L));
% %         At = @(W) sum(Masks .* ifft2(W), 3) * size(W,1) * size(W,2);
% %         % xhat(:,:,aa) = TWFgrad(Z(:,:,aa),Y(:,:,:,aa), Paramstwf, A, At) ;
% %         
% %         grad = compute_grad2(Z(:,:,aa), Y(:,:,:,aa), Paramstwf, A, At);
% %         Z(:,:,aa) =  Z(:,:,aa)- Paramstwf.mu * grad;
% %     end
% %     Xhat          =   reshape(Z,n,q);
% %     %     [UtwfE,SE,VE] =   svds(Xhat,r);
% %     %     DD           =   UtwfE*SE*VE';
% %     [UE,~,~,~] =   BlockIter(Xhat,MaxIter,r);
% %     %[UtwfE,SE,VE] =   svds(Xhat,r);
% %     %DD           =   UtwfE*SE*VE';
% %     DD           =   UE*(UE'*Xhat);
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %     % % %     for na = 1 : q
% %     % % %         xa_hat      =   DD(:,na);
% %     % % %         xa          =   X(:,na);
% %     % % %         %                 %  Tmp_Err_X(na)   =   min(norm(xa-xa_hat)^2, norm(xa+xa_hat)^2);
% %     % % %         Tmp_Err_X(na) 	=   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
% %     % % %     end
% %     % % %     % Rel_Err(:,t)=  Tmp_Err_X;
% %     % % %     Rel_Err      = sum(Tmp_Err_X);
% %     % % %     ERR(:,t)      =   Rel_Err / Den_X;
% %     %%%   fprintf('TWF Projection Error is:\t%2.2e\n',ERR(:,t));
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %     
% %     Z             =   reshape(DD,n_1,n_2,q);
% %     
% % end
% %TimeTWFLoop   =   toc;
% 
% 
% 
% 
% 
