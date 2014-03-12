function [ sentTree, outputItr, innerItr, inputItr ] = buildTree( sentMean, numWords, W, b, U, c, V, d )
%buildTree Builds the tree of the sentence
%   greedy tree algorithm
    
    numNodes = numWords;
    %columns: child 1, child 2, meaning vector
    %0 in child column indicates no child
    nodelist = cell(size(sentMean,2),1);
    for i=1:numWords;
        node=cell(9,1);
        %each node contains the following:
        % 1: meaning vector
        % 2: # leafs under it (for leaf nodes this is 1)
        % The rest are meaningless for leaf nodes
        % 3: z, predicted label
        % 4: a, activation
        % The rest are meaningless for non-output nodes
        % 5: is output (0=false, 1=true)
        % 6: zl vector (predicted meaning vector for left child
        % 7: zr vector (predicted meaning vector for right child
        % 8: RAE error
        % 9: delta vector
        
        node{1} = sentMean(:,i);
        node{2} = 1;%# leafs
        node{5} = 0;%output node?
        nodelist{i} = tree(node);
    end
    
    %build tree of sentence
    while(numNodes>1)
        %inner loop of tree construction: 
        %for each pair of nodes (starting with leaf nodes i.e. the sentence)
        err = ones(numNodes-1,3);
        k = 1;
        for j=1:numNodes-1
            %compute RAE error
            node1=j;
            node2=j+1;

            childi = nodelist{node1}.Node{1};
            childj = nodelist{node2}.Node{1};
            
            xi = childi{1};
            xj = childj{1};

            ni = 1;
            nj = 1;

            xk = meaningFunc(xi,xj,W,b);
            [err(j,1)] = raeError( xk, xi, xj, ni, nj, U, c, d );
            err(j,2) = node1;
            err(j,3) = node2;
            k = k + 1;
        end    %end inner loop
        [val,idx] = min(err(:,1));
        child1Idx = err(idx,2);
        child2Idx = err(idx,3);
        child1 = nodelist{child1Idx}.Node{1};
        child2 = nodelist{child2Idx}.Node{1};
        
        xi = child1{1};
        xj = child2{1};
        [xk,ak] = meaningFunc(xi,xj,W,b);
        [errk, zi, zj] = raeError( xk, xi, xj, ni, nj, U, c, d );
        zk = predictNode(xk,V);
        
        newnode = cell(7,1);
        newnode{1} = xk;
        newnode{2} = child1{2} + child2{2};
        newnode{4} = ak;
        newnode{5} = 0;
        %newnode{3} = zk;
        newnode{6} = zi;
        newnode{7} = zj;
        newnode{8} = errk;

        newtree = tree(newnode);
        newtree = newtree.graft(1,nodelist{child1Idx});
        newtree = newtree.graft(1,nodelist{child2Idx});

        newnodelist = cell(numNodes-1,1);
        for i=1:numNodes
            if i<child1Idx
                newnodelist{i} = nodelist{i};
            elseif i==child1Idx
                newnodelist{i} = newtree;
            elseif i>child2Idx
                newnodelist{i-1} = nodelist{i};
            end
        end
        nodelist=newnodelist;

        numNodes = numNodes - 1;    
    end
    
    sentTree = nodelist{1};
    
    %mark output nodes
    %first go down left nodes until you hit a leaf
    idx = 1;%start at root node
    while (~sentTree.isleaf(idx))
        node = sentTree.get(idx);
        node{5} = 1;
        sentTree = sentTree.set(idx, node);
        childs = sentTree.getchildren(idx);
        idx = childs(1);
    end
    
    %then go down right nodes
    idx = 1;%start at root node
    while (~sentTree.isleaf(idx))
        node = sentTree.get(idx);
        node{5} = 1;
        sentTree = sentTree.set(idx, node);
        childs = sentTree.getchildren(idx);
        idx = childs(2);
    end
        
    outIdx=1;
    iterator = sentTree.breadthfirstiterator;
    for i = iterator
        node = sentTree.get(i);
        if (node{5})
            outputItr(outIdx) = i;
            outIdx = outIdx + 1;
        end
    end
    inputItr=sentTree.findleaves();

    innerItr = sentTree.breadthfirstiterator;
    innerItr = innerItr(~ismember(innerItr,outputItr));
    innerItr = innerItr(~ismember(innerItr,inputItr));

    if 0
        test=sentTree.breadthfirstiterator;
        test1 = sum(ismember(outputItr,test));
        test2 = sum(ismember(innerItr,test));
        test3 = sum(ismember(inputItr,test));
    end

    %disp output nodes for debug
    if 0
        na_order = tree(sentTree, 'clear');
        iterator = sentTree.breadthfirstiterator;
        for i = iterator
            node = sentTree.get(i);
            na_order = na_order.set(i, node{5});
        end
        disp(na_order.tostring);
    end
    %disp nodes idxs for debug
    if 1
        na_order = tree(sentTree, 'clear');
        iterator = sentTree.nodeorderiterator;
        idx = 1;
        for i = iterator
            na_order = na_order.set(i, idx);
            idx = idx + 1;
        end
        disp(na_order.tostring);
    end
end

