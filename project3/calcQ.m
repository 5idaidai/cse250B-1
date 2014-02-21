function [q] = calcQ(words,z,vocabSize,numTopics)
%CALCQ Calculate q matrix in LDA Gibbs training
%   q matrix contains count of each word assigned to topic j in entire
%   corpus

    q = zeros([numTopics,vocabSize]);
    
    for w=1:vocabSize
        widxs=find(words==w);
        for j=1:numTopics
            zidxs=find(z(widxs)==j);
            q(j,w) = size(zidxs,1);
        end
    end

end

