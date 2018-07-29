function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Thet  a2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% EXTRACT NN-PREDICTIONS......

  % COMPUTING HIDDEN LAYER...
  a1 = [ones(m,1) X];				% each row of X = training example + bias unit
  z2 = a1*Theta1';					% input => sigmoid() => hidden label

  % COMPUTING OUTPUT LAYER...
  a2 = [ones(size(z2,1),1) sigmoid(z2)];		% each row = weighted sum of input-units + bias unit
  z3 = a2*Theta2';				
  a3 = sigmoid(z3);					% predictions 

 % FORMAT PREDICTION -> VECTOR
 yVec = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

 % COST FUNCTION
 J = (sum(sum((-yVec).*log(a3) - (1-yVec).*log(1-a3) )))/m;
 reg = (lambda/(2*m)) * ( sum(sum(Theta1(:,2:end)).^2) + sum(sum(Theta2(:,2:end)).^2)  );
 J = J+ reg; 
 J=J;



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients


% Compute low-deltas
d3 = a3-yVec;                                                        % error in (Y)
d2 = (d3*Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)];        % error in hidden layer (a2)


% Compute "ACCUMULATORS" (coeff for Gradients)
D1 = d2(:,2:end)' * a1;    % coeff for Theta1_grad (bias term has no error)
D2 = d3' * a2;             % coeff for Theta2_grad (y' has no bias term)


% Compute Gradient
Theta1_grad = (1/m)*D1;
Theta2_grad = (1/m)*D2;

%Add regularization-term to non-bias units
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) +  (lambda/m)*(Theta2(:,2:end));

Theta1_grad = Theta1_grad;
Theta2_grad = Theta2_grad;

grad = [Theta1_grad(:) ; Theta2_grad(:)];
end