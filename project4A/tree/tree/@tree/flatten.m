function levelContent = flatten(obj)
%% FLATTEN  Return an unordered list of node IDs arranged per level

    %depth = obj.depth + 1;
    [dt, ~, depth] = depthtree(obj);
    depth=depth+1;
    
    levelContent = cell( depth, 1 );
    
    prevLevel = 1;
    levelContent{1} = prevLevel; % Only root at this level
    
    for level = 2 : depth
        arr=cell2mat(dt.Node)';
        test1 = find(arr == level-1);
        levelContent{level} = test1;
    end
    
end