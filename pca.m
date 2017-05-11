function [ X_red, U ] = pca(X)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    [m, n] = size(X);
    Sigma = 1 / m * X' * X;
    [U, S] = svd( Sigma );
    s = S * ones(n, 1);
    total = sum(s);
    Sum = 0;
    for K = 1:n
        Sum = Sum + s(K);
        if Sum / total >= 0.99
            break;
        end
    end
    U = U(1:K, :);
    X_red = X * U';
end

