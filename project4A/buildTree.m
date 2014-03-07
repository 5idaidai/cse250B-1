function [ sentTree ] = buildTree( sentMean, numWords, W, b, U, c, V, d )
%buildTree Builds the tree of the sentence
%   greedy tree algorithm
    
    numNodes = numWords;
    %columns: child 1, child 2, meaning vector
    %0 in child column indicates no child
    nodelist = cell(size(sentMean,2),1);
    for i=1:numWords;
        node=cell(6,1);
        %each node contains the following:
        % 1: meaning vector
        % 2: # leafs under it (for leaf nodes this is 1)
        % The rest are meaningless for leaf nodes
        % 3: label
        % 4: zl vector (predicted meaning vector for left child
        % 5: zr vector (predicted meaning vector for right child
        % 6: RAE error
        
        node{1} = sentMean(:,i);
        node{2} = 1;
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
        xk = meaningFunc(xi,xj,W,b);
        [errk, zi, zj] = raeError( xk, xi, xj, ni, nj, U, c, d );
        %zk = predictNode(xk,V);
        
        newnode = cell(6,1);
        newnode{1} = xk;
        newnode{2} = child1{2} + child2{2};
        %newnode{3} = zk;
        newnode{4} = zi;
        newnode{5} = zj;
        newnode{6} = errk;

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
end

