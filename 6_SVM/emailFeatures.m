function x = emailFeatures(word_indices)

% n-dimensional vector, elements --> existence (1 or 0) of certain word (feature) in email
x = zeros(n, 1);
for i = 1:length(word_indices),		%the list "word_indices" gives indices of found-words
	x(word_indices(i)) = 1;
end
end