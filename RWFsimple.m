function [z] = RWFsimple(y1, Params, A, At)

%% Initialization
npower_iter = Params.npower_iter;           % Number of power iterations
z0 = randn(Params.r,1); z0 = z0/norm(z0,'fro');    % Initial guess
normest = (sqrt(pi/2)*(1-Params.cplx_flag)+sqrt(4/pi)*Params.cplx_flag)*sum(y1(:))/numel(y1(:));
% Estimate norm to scale eigenvector

ytr=y1.* (abs(y1) > 1 * normest );% truncated version
for tt = 1: npower_iter
    z0 = At( ytr.* (A(z0)) ); z0 = z0/norm(z0,'fro');
end
z0 = normest * z0; % Apply scaling

%% reshaped Wirtinger flow
%Relerrs=zeros(Params.T+1,1);
z=z0;
%Relerrs(1) = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error

mu=0.8+0.4*Params.cplx_flag;% real step size 0.8/ complex step size1.2
t=1;
while t<=Params.Tb_LRPRnew
    yz=A(z);
    %   ang   =  Params.cplx_flag*exp(1i * angle(yz)) +(1 - Params.cplx_flag) * sign(yz);
    z = z - mu* (Params.m\At(yz-y1.*yz./abs(yz)));
  %  Relerrs(t+1)=norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro');
    t=t+1;
end
