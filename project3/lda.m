function [ output_args ] = lda(numTopics, counts, vocab)
%LDA performs LDA based training on documents
    counts=counts';
	m=size(counts,2);
	k=numTopics;
	V=size(vocab);
	wordsPerDoc = sum(counts);
    numWords = sum(wordsPerDoc); %total # of words in all docs (so sum of all counts in all docs)
	
    words = get_words(counts, vocab, numWords, m);

	%init alpha & beta -> everything uniformly likely
	alphas = ones(k) * (50/k);
	betas = ones(V) * 0.01;

	%init all z randomly
	z = randi(k,[numWords,1]);
	
	%init n (mxk)
    % count of how many words within doc m are assigned to topic j
	n = randi(k,[m,k]);
	
	%init q (mxk)
	q = calcQ(words,z,V,k);
    
	gibbs(numWords,zs,alphas,betas,counts,n,q);
	
end