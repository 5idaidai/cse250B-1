function [ z, numChanged ] = gibbs(z,words,alphas,betas,n,q,numWords,numTopics,wordsPerDoc)
%GIBBS Summary of this function goes here
%   standard approach to implementing Gibbs sampling iterates over every
% position of every document, taking the positions in some arbitrary order. For
% each position, Equation (5) is evaluated for each alternative topic j. For each j,
% evaluating (5) requires constant time, so the time to perform one epoch of Gibbs
% sampling is O(NK) where N is the total number of words in all documents and
% K is the number of topics.

    numChanged = 0;
    %iterate over every position of every doc (arbitrary order)
    for i=1:numWords
        
        docnum=1;
        totalWords = 0;
        for m=1:size(wordsPerDoc)
            totalWords = totalWords + wordsPerDoc(m);
            if totalWords >= i
                docnum = m;
                break;
            end
        end
        
        oldtopic = z(i);
        word = words(i);
        newtopic = innergibbs(i, word, oldtopic, docnum, alphas, betas, q, n, numTopics);
        
        if newtopic ~= oldtopic
            numChanged = numChanged + 1;
            z(i) = newtopic;
                    
            q(oldtopic,word) = q(oldtopic,word) - 1;
            q(newtopic,word) = q(newtopic,word) + 1;

            n(docnum,oldtopic) = n(docnum,oldtopic) - 1;
            n(docnum,newtopic) = n(docnum,newtopic) + 1;
        end
    end
end

