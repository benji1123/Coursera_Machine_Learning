
function [J, grad] = lrCostFunction(theta, X, y, lambda)

% ====================== VARIABLES ======================

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================


% COST FUNCTION

pred = sigmoid(X*theta);											% model-predictions to be compared w/ label
regTerm = lambda/(2*m)*sum(theta.*2);								% increment cost 
J = ((1/m) * sum((-y)'*log(pred) - (1-y)'*log(1-pred))) + regTerm;	% square difference of label & predictions



% GRADIENT

grad_old = (1/m)*(sum((pred-y).*X));						% first grad value should not be regularized
grad = ((1/m)*(sum((pred-y).*X))) + (lambda/m)*theta';
grad(1)=grad_old(1);
grad = grad(:);
end