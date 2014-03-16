function [ dW, backTreeZ ] =...
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
        
dW = zeros(size(W));
backTreeZ = tree(sentTree, zeros(size(d,1)));

%set delta values for root
node = sentTree.get(1);
children = sentTree.getchildren(1);
childR = sentTree.get(max(children));
childL = sentTree.get(min(children));
a = node{4};
p = node{3};

[ deltaRt] = deltaRoot( t, p );
backTreeZ =  backTreeZ.set(1, deltaRt);

deltaW = deltaRt * [childR{1};childL{1}]';
dW = dW + deltaW;

%set delta values for all autoencoder nodes
for idx=outputItr    

    node = sentTree.get(idx);
    deltak = backTreeZ.get(sentTree.getparent(idx));
    children = sentTree.getchildren(idx);
    childR = sentTree.get(max(children));
    childL = sentTree.get(min(children));
    a = node{1};
    
    parent = sentTree.getparent(idx);
    nodes = sentTree.getchildren(parent);
    nodeL = min(nodes);
    if nodeL==idx
        Wk = W(:,1:d);
    else
        Wk = W(:,d+1:2*d);
    end

    [ delta ] = deltaOutput( a, deltak, Wk );
    backTreeZ = backTreeZ.set(idx, delta);
    
    deltaW = delta .* [childL{1};childR{1}];
    dW = dW + deltaW;  
end

end

