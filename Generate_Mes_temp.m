function [Ysqrt,Y,A] = Generate_Mes_temp(X,Params,m, q)
%$$$$$
% Generating design matrices and measurements
%$$$$$
A      =   zeros(Params.n, m, q);% Design matrices
Y      =   zeros(m, q);% Matrix of measurements
for nl = 1 : q
    A(:, :, nl) = randn(Params.n, m);
    Y(:,nl)     = abs(A(:,:,nl)' * X(:,nl)).^2;
end
%$$$
Ysqrt = sqrt(Y);