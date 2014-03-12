function [ backTreeZ ] = backProp( sentTree, t, outputItr, innerItr, inputItr, U, W, d, deltaP )
%UNTITLED5 Summary of this function goes here
%each node contains the following:
        % 1: meaning vector (zi)
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

backTreeZ = tree(sentTree, 'clear');
backTreeW = tree(sentTree, 'clear');
backTreeU = tree(sentTree, 'clear');

%set delta values for root
node = sentTree.get(1);
children = getchildren(1);
childR = sentTree.get(max(children));
childL = sentTree(min(children));
tr = childR{1};
tl = childL{1};
zl = node{6};
zr = node{7};
a = node{4};
p = {3};
Ul = U(1:d,:);
Ur = U(d+1:2*d,:);

deltaRt = deltaRoot(tl, zl, tr, zr, Ul, Ur, a, t, p);

backTreeZ =  backTreeZ.set(1, deltaRt);

deltaW = deltaP*node{1};
backTreeW = backTreeW.set(1, deltaW);







end

