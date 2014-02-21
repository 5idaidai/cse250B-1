function [words,wordsPerDoc] = get_words(counts, vocab, numWords, numDocs)
%GET_WORDS Summary of this function goes here
%   Detailed explanation goes here

    words = zeros(numWords,1,'uint32');
    
    %m=doc, w=word, c=count
    [w,M,c] = find(counts);
    
    disp('Starting word occurrance calc');
    widx = 1;
    for i=1:size(c)
        for j=1:c(i)
            words(widx) = w(i);
            widx = widx + 1;
        end
    end
end

