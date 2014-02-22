function [ thetas,phis ] = lda(counts, vocab, words, numTopics, numEpochs, percCutOff)
%LDA performs LDA based training on documents
    counts=counts';
	m=size(counts,2);
	k=numTopics;
	V=size(vocab,1);
	wordsPerDoc = sum(counts);
    numWords = sum(wordsPerDoc); %total # of words in all docs (so sum of all counts in all docs)

	%init alpha & beta -> everything uniformly likely
	alphas = ones(k,1) * (50/k);
	betas = ones(V,1) * 0.01;

	%init all z randomly
	z = randi(k,[numWords,1]);
	
	%init n (mxk)
    % count of how many words within doc m are assigned to topic j
	n = nCalc(wordsPerDoc,z,m,k);
	
	%init q (mxk)
	q = calcQ(words,z,V,k);
    
    format shortg
    c=clock;
    fprintf('Starting Gibbs sampling, %d epochs, start time: %d:%d:%.02f\n',numEpochs,c(4),c(5),c(6));

    oldratio = 1;
    for epoch=1:numEpochs
        fprintf('Epoch #%d, ',epoch);
        [z,numChanged] = gibbs(z,words,alphas,betas,n,q,numWords,k,wordsPerDoc);
        
        ratio = numChanged / numWords;
        fprintf('ratio: %f\n',ratio);
        
        if ratio < percCutOff && oldratio < percCutOff
            fprintf('Percentage changed (%f) less than %f',ratio,percCutOff);
        end
        oldratio = ratio;
    end
    c=clock;
    fprintf('Finished Gibbs sampling, end time: %d:%d:%.02f\n',c(4),c(5),c(6));

    
    %recover thetas and phis from learned z distribution
    [thetas,phis] = phiAndTheta(q,n,m,k,V);
	
end