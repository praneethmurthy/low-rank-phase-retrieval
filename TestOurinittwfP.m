clc;
clear;
close all;

%ob = VideoReader('mouse.mp4');
ob = VideoReader('videos/sara.mp4');
vidFrames = read(ob);
%numFrames = get(ob, 'numberOfFrames');
numFrames  =size(vidFrames,4);
for i = 1: numFrames
    temp = vidFrames(:,:,:,i);
    temp = rgb2gray(temp);
    temp = double(temp);
    temp = imresize(temp,0.5);
    % % % %     %if i == 1
    p=size(temp,1);
    d=size(temp,2);
    % % % %     %end
    I(:,i)=reshape(temp,[p*d,1]);
end
% % % %
% n_1     =   15;
n_1     =   p;
% n_2     =   10;
n_2     =   d;
r       =   25;
% q       =   100;
q       =  numFrames ;
MaxIter  =  50;
%%USED%%%%%[U,S,V] = svds(I,r);%%%%%%%%%%%%%%%%%%%%%%%%
%[U2,S2,V2] = svd(I);
%[S U sv ] = hosvd(temp, ones(n_1,n_2,q));
%X  =  U(:,1:r)*U(:,1:r)'*I;
%X2 = U2(:,1:r)*S2(1:r,1:r)*V2(:,1:r)';
%%USED%%%%%X      = U*S*V';%%%%%%%%%%%%%%%%%%%%%%%%%%%
X      =    I;
% X = U*U'*I;
%USED%%%%%%X    =    I;

%Xmat   =  reshape(Xvec,n_1,n_2,q);

L       =   3;
alpha_y =   3;
itr_num_pow_mth    =    30;
m       =   n_1*n_2*L;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n       =    n_1*n_2;
%m       =    n*L;
% q       =    numFrames;
% n_1     =     p;
% n_2     =     d;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%  Alocating input constants for function alt_min_init
Params.n_1     =   n_1;
Params.n_2     =   n_2;
Params.r       =   r;
Params.q       =   q;
Params.L       =   L;
Params.alpha_y =   3;
Params.itr_num_pow_mth    =    itr_num_pow_mth;
Params.m       =   n_1*n_2*L;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Paramstwf.n    =   n;
Paramstwf.n1   =   n_1;
Paramstwf.n2   =   n_2;
%Paramstwf.n  =   size(X,1);
%Paramstwf.n2   =   1;
Paramstwf.cplx_flag   = 0;
Paramstwf.alpha_lb    = 0.3;
Paramstwf.alpha_ub    = 5;
Paramstwf.alpha_h     = 5;
Paramstwf.alpha_y     = 3;
Paramstwf.T           = 50;	% number of iterations
Paramstwf.mu          = 0.2;	% step size / learning parameter
Paramstwf.npower_iter = 50;	% number of power iterations
Paramstwf.grad_type   = 'TWF_Poiss';
Paramstwf.alpha       = q;
Paramstwf.r           = r;
Paramstwf.L           = L;
Paramstwf.m           = m;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Den_X      =   norm(X,'fro');
frm_count  =   0;
XX         =   reshape(X,n_1,n_2,numFrames);

Masks2     =   zeros(n_1,n_2,L,q);
for   a    =   1  :   q
    Masks2(:,:,:,a) = reshape(randsrc(n_1*n_2, L, [1i -1i 1 -1]), n_1, n_2, L);
end

Y            =   zeros(n_1,n_2,L,q);
TWFinithat   =   zeros(n_1,n_2,q);
%timeinit     =   zeros(q,1);
for    a     =  1  :  q
    frame     =    XX(:,:,a);
    frm_count =   frm_count + 1;
    fprintf('Frame # %d ...\n', frm_count);
    
    %     for ll = 1:Paramstwf.L
    %         Masks(:,:,ll) = randsrc(n_1,n_2,[1i -1i 1 -1]);
    %     end
    Masks     =   Masks2(:,:,:,a);
    A  = @(I)  fft2(conj(Masks) .* reshape(repmat(I,[1 Paramstwf.L]), size(I,1), size(I,2), Paramstwf.L));
    At = @(W) sum(Masks .* ifft2(W), 3) * size(W,1) * size(W,2);
    
    Y(:,:,:,a)           =   abs(A(frame)).^2;
end

fprintf('defining complete initializing');
%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
% TmpMask    =    zeros(n_1*n_2,q,L);
%
% for nl = 1 : L
%     TmpMask(:,:,nl) 	=   randsrc(n_1*n_2, q, [1,-1,1i,-1i]);
% end
%
% Masks   =   reshape(TmpMask, n_1,n_2,L,q);
%/////////////////////////////////////////////////
%  Defining functions
Masks1    =  Masks2;
Afull 	 =	@(I) fft2(conj(Masks1).*reshape(repmat(I,[1,L]), n_1, n_2, L, q));
Afull_t  =	@(E) sum(sum( Masks1 .* ifft2(E), 3) , 4)* n_1 * n_2; %* size(E,3);
Afull_tk =	@(S) sum( Masks1 .* ifft2(S), 3)*n_1*n_2;% * size(E,1) * size(E,2) * size(E,3);

%/////////////////////////////////////////////////

%Y       =   abs(Afull(Xmat)).^2;
[Altmintime,Bhat, Uhat,Xhat] = alt_min_init(Y, Params, Afull, Afull_t, Afull_tk);
%Z        =    reshape(Xhat,n_1*n_2,q);


D       =        reshape(Xhat,n,q);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for na = 1 : q
    xa_hat      =   D(:,na);
    xa          =   X(:,na);
    %                 %  Tmp_Err_X(na)   =   min(norm(xa-xa_hat)^2, norm(xa+xa_hat)^2);
    Tmp_Err_X1(na) 	=   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
end
% Rel_Err(:,t)=  Tmp_Err_X;
Err      = sum(Tmp_Err_X1);
ERRinit     =   Err / Den_X;
fprintf('Our initialization Error is:\t%2.2e\n',ERRinit);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Z      =    Xhat;
% % MaxIter  =  50;
% % %tic;
% % [Ut,~,~,~ ]     =    BlockIter(XTWFinit,MaxIter,r);
% % %T2    =  toc;
% % %z              =   Utwf * Utwf' * XTWFinit;
% % %D               =    Utwf*Stwf*Vtwf';
% % D               =    Ut*(Ut'*XTWFinit);
% % Z               =    reshape(D,n_1,n_2,q);
% % %Tsvd            =    toc;
% % %Timeinittot     =    Timeinit+Tsvd;
% % TT              =    toc;
tic;
for t = 1: Paramstwf.T
    fprintf('Iteration # %d of main loop...\n', t);
    for   aa   =   1  :  q
        Masks  =   Masks2(:,:,:,aa);
        A  = @(I)  fft2(conj(Masks) .* reshape(repmat(I,[1 Paramstwf.L]), size(I,1), size(I,2), Paramstwf.L));
        At = @(W) sum(Masks .* ifft2(W), 3) * size(W,1) * size(W,2);
        % xhat(:,:,aa) = TWFgrad(Z(:,:,aa),Y(:,:,:,aa), Paramstwf, A, At) ;
        
        grad = compute_grad2(Z(:,:,aa), Y(:,:,:,aa), Paramstwf, A, At);
        Z(:,:,aa) =  Z(:,:,aa)- Paramstwf.mu * grad;
    end
    Xhat          =   reshape(Z,n,q);
    %     [UtwfE,SE,VE] =   svds(Xhat,r);
    %     DD           =   UtwfE*SE*VE';
    [UE,~,~,~] =   BlockIter(Xhat,MaxIter,r);
    %[UtwfE,SE,VE] =   svds(Xhat,r);
    %DD           =   UtwfE*SE*VE';
    DD           =   UE*(UE'*Xhat);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % %     for na = 1 : q
    % % %         xa_hat      =   DD(:,na);
    % % %         xa          =   X(:,na);
    % % %         %                 %  Tmp_Err_X(na)   =   min(norm(xa-xa_hat)^2, norm(xa+xa_hat)^2);
    % % %         Tmp_Err_X(na) 	=   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
    % % %     end
    % % %     % Rel_Err(:,t)=  Tmp_Err_X;
    % % %     Rel_Err      = sum(Tmp_Err_X);
    % % %     ERR(:,t)      =   Rel_Err / Den_X;
    %%%   fprintf('TWF Projection Error is:\t%2.2e\n',ERR(:,t));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Z             =   reshape(DD,n_1,n_2,q);
    
end
TimeTWFLoop   =   toc;
vdo_out_obj =   VideoWriter('AMT2PlaneOrigR25L3');
open(vdo_out_obj);
Tmp_Err_X2   =   zeros(q, 1);
for   t    =  1  :   q
    writeVideo(vdo_out_obj, uint8(abs(Z(:,:,t))));
    xa_hat        =   DD(:,t);
    xa            =   X(:,t);
    Tmp_Err_X2(t)  =   norm(xa - exp(-1i*angle(trace(xa'*xa_hat))) * xa_hat, 'fro');
end
Nom_Err_X_twf	    =   sum(Tmp_Err_X2);
ERRTWFP             =  Nom_Err_X_twf / Den_X;
close(vdo_out_obj);


