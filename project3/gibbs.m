function [ output_args ] = gibbs(numWords,zs,alphas,betas,counts,n,q)
%GIBBS Summary of this function goes here
%   standard approach to implementing Gibbs sampling iterates over every
% position of every document, taking the positions in some arbitrary order. For
% each position, Equation (5) is evaluated for each alternative topic j. For each j,
% evaluating (5) requires constant time, so the time to perform one epoch of Gibbs
% sampling is O(NK) where N is the total number of words in all documents and
% K is the number of topics.

    %iterate over every position of every doc (arbitrary order)
	%for i=1tonumWords
		%for k=1toK -> this might not be necessary depending on if we use matrix magic for inner gibbs
			%innergibbs(alpha, beta, q, n, k)

end

