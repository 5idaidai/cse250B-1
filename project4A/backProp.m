function [ dW,dU,dV, backTreeZ, dMeaning ] =...
    backProp( sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, V, d, alpha, trainInput )
%UNTITLED5 Summary of this function goes here
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
        
dV = zeros(size(V));
dW = zeros(size(W));
dU = zeros(size(U));
if trainInput
    dMeaning = sparse(size(meanings,1),size(meanings,2));
end
backTreeZ = tree(sentTree, zeros(size(d,1)));

%set delta values for root
node = sentTree.Node{1};
children = sentTree.getchildren(1);
childR = sentTree.Node{max(children)};
childL = sentTree.Node{min(children)};
tr = [childR{1};1];
tl = [childL{1};1];
zl = [node{6};1];
zr = [node{7};1];
a = node{4};
p = node{3};
Ul = U(1:d,:);
Ur = U(d+1:2*d,:);
topL = childL{2};
topR = childR{2};
bottom = topL+topR;

[ deltaRt, deltaP, dLdGammaL, dLdGammaR ] = deltaRoot( tl, zl, tr, zr, Ul, Ur, a, t, p, V, topL, topR, bottom, alpha);
backTreeZ =  backTreeZ.set(1, deltaRt);

deltaW = deltaRt * [childL{1};childR{1};1]';
dW = dW + deltaW;

deltaV = deltaP*node{1}';
dV = dV + deltaV;

deltaU = buildDeltaU(node{1},dLdGammaL,dLdGammaR);
dU = dU + deltaU;

%set delta values for all autoencoder nodes
for idx=outputItr    

    node = sentTree.Node{idx};
    deltak = backTreeZ.get(sentTree.getparent(idx));
    children = sentTree.getchildren(idx);
    childR = sentTree.Node{max(children)};
    childL = sentTree.Node{min(children)};
    tr = [childR{1};1];
    tl = [childL{1};1];
    zl = [node{6};1];
    zr = [node{7};1];
    a = node{4};
    p = node{3};
    Ul = U(1:d,:);
    Ur = U(d+1:2*d,:);
    topL = childL{2};
    topR = childR{2};
    bottom = topL+topR;
    
    parent = sentTree.getparent(idx);
    nodes = sentTree.getchildren(parent);
    nodeL = min(nodes);
    if nodeL==idx
        Wk = W(:,1:d);

    else
        Wk = W(:,d+1:2*d);

    end

    [ delta, deltaP, dLdGammaL, dLdGammaR ] = deltaOutput( t, p, tl, zl, tr, zr, deltak, Wk, Ul, Ur, a, V, topL, topR, bottom, alpha);
    backTreeZ = backTreeZ.set(idx, delta);
    
    deltaW = delta * [childL{1};childR{1};1]';
    dW = dW + deltaW;

    deltaV = deltaP*node{1}';
    dV = dV + deltaV;

    deltaU = buildDeltaU(node{1},dLdGammaL,dLdGammaR);
    dU = dU + deltaU;    
end

%set delta values for all inner nodes
for idx=innerItr    

    node = sentTree.Node{idx};
    paridx = sentTree.getparent(idx);
    deltak = backTreeZ.get(paridx);
    a = node{4};
    p = node{3};
    
    parent = sentTree.getparent(idx);
    nodes = sentTree.getchildren(parent);
    nodeL = min(nodes);
    if nodeL==idx
        Wk = W(:,1:d);
    else
        Wk = W(:,d+1:2*d);
    end
    
    [ delta, deltaP ] = deltaNonOutput( a, deltak, Wk, V, t, p, alpha );
    backTreeZ = backTreeZ.set(idx, delta);
    
    deltaW = delta * [childL{1};childR{1};1]';
    dW = dW + deltaW;

    deltaV = deltaP*node{1}';
    dV = dV + deltaV;
end

if trainInput
    %set delta values for all leaf nodes
    for idx=inputItr

        node = sentTree.Node{idx};
        deltak = backTreeZ.get(sentTree.getparent(idx));
        a = node{4};
        p = node{3};

        parent = sentTree.getparent(idx);
        nodes = sentTree.getchildren(parent);
        nodeL = min(nodes);
        if nodeL==idx
            Wk = W(:,1:d);
        else
            Wk = W(:,d+1:2*d);
        end

        [ delta, deltaP ] = deltaInput( a, deltak, Wk, V, t, p, alpha );
        deltaV = deltaP*node{1}';
        dV = dV + deltaV;
        
        wordIdx=node{12};
        tempM=dMeaning';
        [i,j,k] = find(tempM);
        if ismember(wordIdx,i)
            tempM(wordIdx,:) = tempM(wordIdx,:) + delta';
        else
            newrows=[1:20]';
            newcol=ones(20,1)*wordIdx;
            newvals=delta;
            newi = [i;newcol];
            newj = [j;newrows];
            newk = [k;newvals];
            totalVals = nnz(tempM) + length(newvals);
            tempM = sparse(newi,newj,newk,size(tempM,1),size(tempM,2),totalVals);
        end
        dMeaning=tempM';
    end
end

end

