tmperr_qr1 = zeros(100,1);
tmperr_old1 = zeros(100,1);
k = 90;

high_prob_out = zeros(2, 10);
for jj = 1 : 10
    for ii = 1 : 100
        tmperr_qr1(ii) = err_X_iter(1, jj, ii);
        tmperr_old1(ii) = err_X_iter(2, jj, ii);
    end
    qr1 = mink(tmperr_qr1, k);
    old1 = mink(tmperr_old1, k);
    val_qr = mean(qr1);
    val_old = mean(old1);
    high_prob_out(:, jj) = [val_qr; val_old];
end


tmperr = zeros(1,100);
tmperr_large = zeros(1,100);
k = 90;
high_prob_out = zeros(1, 24);
high_prob_out_large = zeros(1, 26);
for jj = 1 : 24
    tmperr = err_SE_iter(jj,:);
    qr1 = mink(tmperr, k);
    val = mean(qr1);
    high_prob_out(jj) = val;
end

for jj = 1 : 26
    tmperr_large = err_SE_iter_large(jj,:);
    qr1 = mink(tmperr_large, k);
    val = mean(qr1);
    high_prob_out_large(jj) = val;
end

