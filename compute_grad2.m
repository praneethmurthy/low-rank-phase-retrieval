
function grad = compute_grad2(z, Y, Params, A, At)
m = Params.m;
Bz = A(z);
absBz = abs(Bz);

normz = norm(z,'fro');
hz_norm = 1/m* norm(absBz(:).^2 - Y(:), 1);
diff_Bz_Y = absBz.^2 - Y;

E    =  (absBz  <= Params.alpha_ub * normz) .* ...
    (absBz  >= Params.alpha_lb * normz) .* ...
    (abs(diff_Bz_Y) <= Params.alpha_h * hz_norm / normz * absBz);
C    = 2* (diff_Bz_Y) ./ conj(Bz)  .*  E;
grad = At(C) / numel(C);                    % Wirtinger gradient
