function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 


m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));


% regularize COST

h = sigmoid(X*theta);							% probability vector
regTerm = lambda/(2*m)*sum(theta.*2);			% increment cost 
J = (1/m) * sum((-y)'*log(h) - (1-y)'*log(1-h)) + regTerm



% regularize GRADIENT

grad_old = (1/m)*(sum((h-y).*X));						% first grad value should not be regularized
grad = (1/m) * sum((h-y).*X) + (lambda/m)*theta'
grad(1)=grad_old(1);

end