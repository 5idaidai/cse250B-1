function [ posidxs, poswords, negidxs, negwords ] = top10Words( words, meanings, V )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    posidxs = zeros(10,2);
    negidxs = zeros(10,2);
    
    minpos = .5;
    
    numWords = length(words);
    
    if size(V,1) == 1
        negidxs = ones(10,2);
        for i=1:numWords
            x = meanings(:,i);
            p = predictNode(x, V); %2x1

            [val,idx]=min(posidxs(:,1));
            if p(1) > minpos && p(1) > val
                posidxs(idx,1) = p(1);
                posidxs(idx,2) = i;
            end

            [val,idx]=max(negidxs(:,1));
            if p(1) < minpos && p(1) < val
                negidxs(idx,1) = p(1);
                negidxs(idx,2) = i;
            end
        end
    elseif size(V,1)==2
        for i=1:numWords
            x = meanings(:,i);
            p = predictNode(x, V);

            [val,idx]=min(posidxs(:,1));
            if p(1) > val
                posidxs(idx,1) = p(1);
                posidxs(idx,2) = i;
            end

            [val,idx]=min(negidxs(:,1));
            if p(2) > val
                negidxs(idx,1) = p(2);
                negidxs(idx,2) = i;
            end
        end        
    end
    
    poswords = words(posidxs(:,2));
    negwords = words(negidxs(:,2));

end

