function [levelContent,depth] = flatten(obj)
%% FLATTEN  Return an unordered list of node IDs arranged per level

    %depth = obj.depth + 1;
    [dt, ~, depth] = depthtree(obj);
    dloc=depth+1;
    
    levelContent = cell( dloc, 1 );
    
    prevLevel = 1;
    levelContent{1} = prevLevel; % Only root at this level
    
    for level = 2 : dloc
        arr=cell2mat(dt.Node)';
        test1 = find(arr == level-1);
        levelContent{level} = test1;
    end
    
end