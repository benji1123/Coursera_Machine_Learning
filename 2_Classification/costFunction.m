function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.


m = length(y); 					% number of training examples

% TEST ACCURACY OF HYPOTHESES (Cost Function) 
h = sigmoid(X*theta);									% probability vector
J = (1/m) * sum((-y)'*log(h) - (1-y)'*log(1-h) )		% error return

% GRADIENT (no idea why)
grad = (1/m) * sum((h-y).*X)							% vector of partial derivativses