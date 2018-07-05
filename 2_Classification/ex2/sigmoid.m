
function g = sigmoid(z)
g = zeros(size(z));

%Compute sigmoid of each value of z (z = matrix, vector or scalar).

rows = size(z,1); 
cols = size(z,2);

for r = 1:rows,
	for c = 1:cols,
	g(r,c) = 1/(1+exp(-z(r,c)));	%non-vectorized
	end;
end;
end