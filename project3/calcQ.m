function [q] = calcQ(words,z,vocabSize,numTopics)
%CALCQ Calculate q matrix in LDA Gibbs training
%   q matrix contains count of each word assigned to topic j in entire
%   corpus

    q = zeros([numTopics,vocabSize]);
    
    for w=1:vocabSize
        widxs=find(words==words(w));
        widxs2=widxs(find(widxs~=w));
        temp=z(widxs2);
        for j=1:numTopics
            q(j,w) = sum(temp==j);
        end
    end

end

