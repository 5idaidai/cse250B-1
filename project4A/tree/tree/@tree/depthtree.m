function [dt, dtiter] = depthtree(obj)
%% DEPTHTREE  Create a coordinated tree where each node holds its depth.
% As for tree.getDepth, the node root has a depth of 0, its children a
% depth of 1, and recursively to the leaves.

    dt = tree(obj,'clear');
    dt.Node{1} = 0;
    
    iterator = obj.depthfirstiterator;
    dtiter = iterator;
    iterator(1) = []; % Remove root
    
    for i = iterator
        
        parent = dt.Parent(i);
        parentDepth = dt.Node{parent};
        dt.Node{i}=parentDepth+1;
        
    end

end