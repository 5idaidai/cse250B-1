function [ thetas,phis ] = lda(numTopics, counts, vocab, numEpochs, percCutOff)
%LDA performs LDA based training on documents
    counts=counts';
	m=size(counts,2);
	k=numTopics;
	V=size(vocab);
	wordsPerDoc = sum(counts);
    numWords = sum(wordsPerDoc); %total # of words in all docs (so sum of all counts in all docs)
	
    words = get_words(counts, vocab, numWords, m);

	%init alpha & beta -> everything uniformly likely
	alphas = ones(k,1) * (50/k);
	betas = ones(V) * 0.01;

	%init all z randomly
	z = randi(k,[numWords,1]);
	
	%init n (mxk)
    % count of how many words within doc m are assigned to topic j
	n = nCalc(wordsPerDoc,z,m,k);
	
	%init q (mxk)
	q = calcQ(words,z,V,k);
    
    fprintf('Starting Gibbs sampling, %d epochs\n',numEpochs);
    for epoch=1:numEpochs
        [z,numChanged] = gibbs(z,words,alphas,betas,n,q,numWords,k,wordsPerDoc);
        
        ratio = numChanged / numWords;
        
        if ratio < percCutOff
            fprintf('Percentage changed less than %f',percCutOff);
        end
    end
    disp('Finished Gibbs sampling');
    
    %recover thetas and phis from learned z distribution
    [thetas,phis] = phiAndTheta(q,n,m,k,V);
	
end