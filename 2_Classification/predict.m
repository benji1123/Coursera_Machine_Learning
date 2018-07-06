function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)


m = size(X, 1); 		% Number of training examples
p = zeros(m, 1);
predictions = sigmoid(X*theta); 	% probabilities used for final estimate

% Simplify model to binary prediction 
for i = 1:m,
	if predictions(i,1) >= 0.5,
	p(i,1) = 1;
	else 
		p(i,1) = 0;
	end
end

predictions = p 	% display output
end