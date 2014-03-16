function [ numDiffTree ] = fwdPropNumDiff( outputItr, innerItr, sentTree, W, U, V, d, t, alpha )
%UNTITLED3 Summary of this function goes here
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
        % 12: word index (only for input nodes)
        % 13: log loss (E2)

    numDiffTree = sentTree;

    %Start at the level above the leaves and work toward the root
    %depth = sentTree.depth;
    [flat,depth] = sentTree.flatten;
    for i=depth:-1:1
        level = flat{i,1};
        idx=level;
        %Iterate through inner nodes first
        iitr=innerItr(ismember(innerItr,idx));
        for j=iitr
            node = sentTree.Node{j};
            children = sentTree.getchildren(j);
            childr = sentTree.Node{max(children)};
            childl = sentTree.Node{min(children)};

            xl = childl{1};
            xr = childr{1};

            nl = childl{2};
            nr = childr{2};

            [xk,ak] = meaningFunc(xl,xr,W);

            [ E1, zl, zr, el, er ] = raeError( xk, xl, xr, nl, nr, U, d, alpha );
            pk = predictNode(xk,V);
            E2 = predError(t,pk,alpha);

            node{1} = xk;
            node{2} = nl + nr;
            node{3} = pk;
            node{4} = ak;
            node{5} = 0;
            node{6} = zl;
            node{7} = zr;
            node{8} = E1;
            node{10} = el;
            node{11} = er;
            node{13} = E2;
            
            numDiffTree = numDiffTree.set(j,node);
        end
        
        %Iterate through outer nodes
        oitr=outputItr(ismember(outputItr,idx));
        for k=oitr
            node = sentTree.Node{k};
            children = sentTree.getchildren(k);
            childr = sentTree.Node{max(children)};
            childl = sentTree.Node{min(children)};

            xl = childl{1};
            xr = childr{1};

            nl = childl{2};
            nr = childr{2};

            [xk,ak] = meaningFunc(xl,xr,W);

            [ E1, zl, zr, el, er ] = raeError( xk, xl, xr, nl, nr, U, d, alpha );
            pk = predictNode(xk,V);
            E2 = predError(t,pk,alpha);

            node{1} = xk;
            node{2} = nl + nr;
            node{3} = pk;
            node{4} = ak;
            node{5} = 1;
            node{6} = zl;
            node{7} = zr;
            node{8} = E1;
            node{10} = el;
            node{11} = er;
            node{13} = E2;

            numDiffTree = numDiffTree.set(k,node);
        end
        
        if i==1 %at root level
            k = 1;
            node = sentTree.Node{k};
            children = sentTree.getchildren(k);
            childr = sentTree.Node{max(children)};
            childl = sentTree.Node{min(children)};

            xl = childl{1};
            xr = childr{1};

            nl = childl{2};
            nr = childr{2};

            [xk,ak] = meaningFunc(xl,xr,W);

            [ E1, zl, zr, el, er ] = raeError( xk, xl, xr, nl, nr, U, d, alpha );
            pk = predictNode(xk,V);
            E2 = predError(t,pk,alpha);

            node{1} = xk;
            node{2} = nl + nr;
            node{3} = pk;
            node{4} = ak;
            node{5} = 1;
            node{6} = zl;
            node{7} = zr;
            node{8} = E1;
            node{10} = el;
            node{11} = er;
            node{13} = E2;

            numDiffTree = numDiffTree.set(k,node);
        end
    end
end




