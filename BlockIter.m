function [U, Sigma, V, iter] = BlockIter( A, MaxIter, k)

[n, m] = size(A);
V = randn(m,k);
iter = 0;
%converge =0;
%Sigma_old=zeros(k,1);
while iter<MaxIter
    [U,~] = qr(A*V, 0);
   
    [V,R] = qr(A'*U, 0);
    Sigma = abs((R));
    %if norm(Sigma-Sigma_old)<1e-4
     %   converge=1;
    %end
    iter = iter + 1;
    %Sigma_old =Sigma;
end


end

