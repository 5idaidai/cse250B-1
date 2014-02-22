function [idx,val] = printTopKWords(phis,vocab,k)
%UNTITLED3 Print top k words in each topic
%   Detailed explanation goes here

    for i=1:size(phis,1)
        topic=phis(i,:);
        [j,idx,val] = find(topic);
        [sval,sidx]=sort(val(:),'descend');
        
        fprintf('Topic %d\n',i);
        for j=1:k
            cell=vocab(sidx(j));
            fprintf('%s\n', cell{1});
        end
        fprintf('\n');
    end

end

