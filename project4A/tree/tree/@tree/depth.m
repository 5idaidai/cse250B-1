function d = depth(obj)
%%DEPTH  Return the depth of the tree. 
% Return 0 if this tree has only a root, 1 if the root has children and
% these children has no chidren, etc...
% The depth of a tree can be seen as the maximum number of edges between
% any leaf and the root.

    [dt,~,d] = obj.depthtree;
    
end