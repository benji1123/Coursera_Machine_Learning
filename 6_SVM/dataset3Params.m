function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 100;
sigma = 0.3;


best_guess = 1000000000; 

%Compute best C-Sigma-Pair from pre-selected list 
for C_i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
	for sigma_i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],

		model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i)); %using "svmTrain.m" model-creation script
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		if error < best_guess,
			best_guess = error; sigma = sigma_i; C=C_i;
		end
	end
end
end