function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)


% GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha


% Initialize some useful values
m = length(y);                          % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================  

        c = alpha/m;

        theta = theta - c * sum((X*theta-y).*X)';
        % source: https://stackoverflow.com/questions/20735406/vectorization-of-a-gradient-descent-code

        %{ NON-VECTORIZED Algorithm
        % EVERY ITERATION TAKES US CLOSER TO THE LOCAL MINIMUM
        t0 = theta(1,1) - c * sum((X*theta-y).*X(:,1));                 % theta(1) decrement  
        t1 = theta(2,1) - c * sum((X*theta-y).*X(:,2));                 % theta(2) decrement

        % SIMULTANEOUS UPDATE 
        theta(1,1) = t0;
        theta(2,1) = t1;
        theta=theta
        %}

    % ============================================================


end

