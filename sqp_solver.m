function [x_opt, lambda_opt, fval, exitflag, output] = sqp_solver(f, gradf, hessf, g, gradg, hessg, x0, options)
    % SQP solver with Quasi-Newton Hessian approximation using modified BFGS updating
    % Inputs:
    %   f       - objective function
    %   gradf   - gradient of objective function
    %   hessf   - Hessian of objective function (unused in Quasi-Newton approach)
    %   g       - constraint function
    %   gradg   - gradient of constraint function
    %   hessg   - Hessian of constraint function (unused in Quasi-Newton approach)
    %   x0      - initial guess
    %   options - optimization options (e.g., max iterations, tolerance, etc.)
    
    % Initialize variables
    x = x0;
    lambda = zeros(length(g(x0)), 1); % Initial Lagrange multipliers
    H = eye(length(x0)); % Initial Hessian approximation
    
    % Extract options
    maxIter = options.MaxIter;
    tol = options.TolFun;
    
    for k = 1:maxIter
        % Step 1: Solve QP subproblem
        [p, lambda_new] = solve_qp_subproblem(H, gradf(x), gradg(x), g(x));
        
        % Step 2: Line search for step size
        alpha = line_search(f, g, x, p, gradf);
        
        % Step 3: Update variables
        x_new = x + alpha * p;
        
        % Check for convergence
        if norm(x_new - x) < tol
            break;
        end
        
        % Update Hessian approximation using modified BFGS
        s = x_new - x;
        y = gradf(x_new) + gradg(x_new)' * lambda_new - (gradf(x) + gradg(x)' * lambda);
        H = bfgs_update(H, s, y);
        
        % Update x and lambda
        x = x_new;
        lambda = lambda_new;
    end
    
    x_opt = x;
    lambda_opt = lambda;
    fval = f(x);
    exitflag = 1;
    output.iterations = k;
end

function [p, lambda] = solve_qp_subproblem(H, gradf, gradg, g)
    % Solve QP subproblem using KKT conditions
    % min 0.5 * p' * H * p + gradf' * p
    % s.t. gradg * p + g = 0
    
    A = gradg;
    b = -g;
    
    KKT_matrix = [H, A'; A, zeros(size(A, 1))];
    KKT_rhs = [-gradf; b];
    
    sol = KKT_matrix \ KKT_rhs;
    p = sol(1:length(gradf));
    lambda = sol(length(gradf) + 1:end);
end

function alpha = line_search(f, g, x, p,gradf)
    % Inexact line search to determine step size
    alpha = 1;
    c1 = 1e-4;
    rho = 0.9;
    
    while f(x + alpha * p) > f(x) + c1 * alpha * gradf(x)' * p || any(g(x + alpha * p) > 0)
        alpha = rho * alpha;
    end
end

function H_new = bfgs_update(H, s, y)
    % Modified BFGS update
    rho = 1 / (y' * s);
    V = eye(length(s)) - rho * s * y';
    H_new = V' * H * V + rho * (s * s');
end
