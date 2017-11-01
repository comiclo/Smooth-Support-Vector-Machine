function y = kernel_rbf(gamma, A, B)
% K(A, B) = \exp(-\gamma * || A - B ||^2)
m = size(A, 1);
n = size(B, 1);
norm_square = repmat(sum(B.^2, 2)', m, 1) - 2*A*B' + repmat(sum(A.^2, 2), 1, n);
y = exp(-gamma * norm_square);
end
