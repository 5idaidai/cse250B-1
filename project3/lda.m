function [ output_args ] = lda(numTopics, counts, vocab)
%LDA performs LDA based training on documents

	m=size(counts);
	k=numTopics;
	V=size(vocab);

	%init alpha & beta -> everything uniformly likely
	alphas = ones(k)/k;
	betas = ones(V)/V;

	numWords = sum(counts); %total # of words in all docs (so sum of all counts in all docs)
	
	%init all z randomly
	z = randi(k,[numWords,1]);
	
	%init n (mxk)
	n = randi(k,[m,k]);
	
	%init q (mxk)
	q = randi(k,[k,numWords]);

	gibbs(numWords,zs,alphas,betas,counts,n,q);
	
end