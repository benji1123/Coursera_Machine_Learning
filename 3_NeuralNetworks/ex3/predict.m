function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% ====================== Resource Variables ===================
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================


% COMPUTING LABEL of HIDDEN LAYER
x1 = [ones(m,1) X];						% each row = training example + bias unit
Theta1=Theta1';							% configure for multiplication with x1
z2 = x1*Theta1;							% input ==> sigmoid() ==> a2 label
a2 = sigmoid(z2);
		

% COMPUTING LABEL OF OUTPUT LAYER
a2 = [ones(size(a2,1),1) a2];			% each row = weighted sum of input-units + bias unit
Theta2 = Theta2';
z3 = a2*Theta2;				
a3 = sigmoid(z3)						% generate test-unit's match with each class


% DISPLAYING MACHINE'S PREDICTIONS		% rows of a3 correspond to diff test-units
for ex = 1:size(a3,1),
	[row,col] = max(a3(ex,:));			% col# => class#, (e.g. 4th col = 4)
	p(ex,1) = col;						% p is return-vector
end
end