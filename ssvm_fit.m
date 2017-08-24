function [ model ] = ssvm_fit(x, y, c)
[m, n] = size(x);
x1 = [x, ones(m, 1)];
a_0 = 1e-5;
w_0 = rand(n + 1, 1) * 2 - 1;

tol = 1e-6;
max_iter = 25;
delta = 0.5; % in (0, 1)
sigma = 0.25; % in (0, 1/2)
tau = 0.5; % in (0, 1)

e_0 = [1; zeros(n + 1, 1)];

a_k = a_0;
w_k = w_0;

G = @(w, a, x, y) norm(H(w, a, x, y, c))^2;
it = 0;
for it=1:max_iter
    H_k = H(w_k, a_k, x1, y, c);
    H_k_norm = norm(H_k);

    if H_k_norm < tol, break; end

    % compute direction
    beta = tau * min(1, H_k_norm^2);
    sol = H_prime(w_k, a_k, x1, y, c, n) \ (a_0 * beta * e_0 - H_k);
    d_a = sol(1);
    d_w = sol(2:end);

    % line search
    theta_k = 1;
    while G(w_k + theta_k * d_w, a_k + theta_k * d_a, x1, y) ...
          > (1 - 2 * sigma * (1 - tau * a_0) * theta_k) * H_k_norm ^ 2
        theta_k = theta_k * delta;
    end

    % update
    a_k = a_k + theta_k * d_a;
    w_k = w_k + theta_k * d_w;
end
model.norm = H_k_norm;
model.w = w_k;
model.it = it;
end

function y = H(w, a, X1, Y, c)
r = 1 - Y .* (X1 * w);
y = [a; - c * sum(sparse(diag(f(r, a) .* Y)) * X1, 1)' + w];
end

function y = H_prime(w, a, X1, Y, c, n)
r = 1 - Y .* (X1 * w);
E = - c * sum(sparse(diag(f_a(r, a) .* Y)) * X1, 1)';
D = c * X1' * (sparse(diag(f_x(r, a))) * X1);
y = [1  zeros(1, n + 1);
     E  D + eye(n + 1)];
end

function y = f(x, a)
r = (x >= a);
y = r .* x + ...
    ~r .* (x > -a) .* (x + a).^2 ./ (4.*a);
end

function y = f_x(x, a)
r = (x >= a);
y = r .* 1 + ...
    ~r .* (x > -a) .* ((x + a) ./ (2 .* a));
end

function y = f_a(x, a)
y = (x < a) .* (x > -a) .* (x + a) .* (a - x) ./ (2 .* a);
end
