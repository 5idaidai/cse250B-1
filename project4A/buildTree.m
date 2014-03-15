function [ sentTree, outputItr, innerItr, inputItr ] ...
        = buildTree( sent, wordMeanings, numWords, W, U, V, d, t, alpha )
%buildTree Builds the tree of the sentence, 
% doing the feed foward calcs at the same time
%   uses greedy tree algorithm
    numCells = 13;
    numNodes = numWords;
    %columns: child 1, child 2, meaning vector
    %0 in child column indicates no child
    nodelist = cell(size(sent,2),1);
    for i=1:numWords;
        node=cell(numCells,1);
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
        % 8: RAE error (E1)
        % 9: delta vector
        % 10: el error for left child on output nodes
        % 11: er error for right child on output nodes
        % only for input nodes
        % 12: word index
        % 13: log loss (E2)
        
        node{12} = sent(i);
        node{1} = wordMeanings(:,node{12});
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

            childl = nodelist{node1}.Node{1};
            childr = nodelist{node2}.Node{1};
            
            xl = childl{1};
            xr = childr{1};

            nl = childl{2};
            nr = childr{2};

            xk = meaningFunc(xl,xr,W);
            [err(j,1)] = raeError( xk, xl, xr, nl, nr, U, d );
            err(j,2) = node1;
            err(j,3) = node2;
            k = k + 1;
        end    %end inner loop
        [val,idx] = min(err(:,1));
        childlIdx = err(idx,2);
        childrIdx = err(idx,3);
        childl = nodelist{childlIdx}.Node{1};
        childr = nodelist{childrIdx}.Node{1};
        
        xl = childl{1};
        xr = childr{1};
        nl = childl{2};
        nr = childr{2};
        [xk,ak] = meaningFunc(xl,xr,W);
        [errk, zl, zr, el, er] = raeError( xk, xl, xr, nl, nr, U, d );
        pk = predictNode(xk,V);
        ll = logLoss(t,pk);

        newnode = cell(numCells,1);
        newnode{1} = xk;
        newnode{2} = nl + nr;
        newnode{3} = pk;
        newnode{4} = ak;
        newnode{5} = 0;
        newnode{6} = zl;
        newnode{7} = zr;
        newnode{8} = errk;
        newnode{10} = el;
        newnode{11} = er;
        newnode{13} = ll;

        newtree = tree(newnode);
        newtree = newtree.graft(1,nodelist{childlIdx});
        newtree = newtree.graft(1,nodelist{childrIdx});

        newnodelist = cell(numNodes-1,1);
        for i=1:numNodes
            if i<childlIdx
                newnodelist{i} = nodelist{i};
            elseif i==childlIdx
                newnodelist{i} = newtree;
            elseif i>childrIdx
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
       
    %build output, inner, input node iterators
    outIdx=1;
    outputItr = [];
    iterator = sentTree.breadthfirstiterator;
    for i = iterator
        node = sentTree.get(i);
        if (i~=1 && node{5})
            outputItr(outIdx) = i;
            outIdx = outIdx + 1;
        end
    end
    inputItr=sentTree.findleaves();

    innerItr = sentTree.breadthfirstiterator;
    innerItr = innerItr(~ismember(innerItr,1));
    innerItr = innerItr(~ismember(innerItr,outputItr));
    innerItr = innerItr(~ismember(innerItr,inputItr));

    %debug code
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
    if 0
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

