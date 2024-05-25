% Example usage
f = @(x) x(1)^2 + x(2)^2; % Objective function
gradf = @(x) [2*x(1); 2*x(2)]; % Gradient of the objective function
g = @(x) x(1) + x(2) - 1; % Constraint function
gradg = @(x) [1; 1]'; % Gradient of the constraint function
x0 = [0.5; 0.5]; % Initial guess
options = struct('MaxIter', 100, 'TolFun', 1e-6); % Options

[x_opt, lambda_opt, fval, exitflag, output] = sqp_solver(f, gradf, [], g, gradg, [], x0, options);
disp('Optimal solution:');
disp(x_opt);
disp('Optimal Lagrange multipliers:');
disp(lambda_opt);
disp('Objective function value at optimal solution:');
disp(fval);
