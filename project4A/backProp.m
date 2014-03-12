function [ backTreeZ ] = backProp( sentTree, t, outputItr, innerItr, inputItr, U, W, d )
%UNTITLED5 Summary of this function goes here
        %each node contains the following:
        % 1: meaning vector (zi)
        % 2: # leafs under it (for leaf nodes this is 1)
        % The rest are meaningless for leaf nodes
        % 3: p, predicted label
        % 4: a, activation
        % The rest are meaningless for non-output nodes
        % 5: is output (0=false, 1=true)
        % 6: zl vector (predicted meaning vector for left child
        % 7: zr vector (predicted meaning vector for right child
        % 8: RAE error
        % 9: delta vector
        % 10: el error for left child on output nodes
        % 11: er error for right child on output nodes

backTreeZ = tree(sentTree, 'clear');
backTreeV = tree(sentTree, 'clear');
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

[ deltaRoot, deltaP, dLdGammaL, dLdGammaR ] = deltaRoot( tl, zl, tr, zr, Ul, Ur, a, t, p, V );
backTreeZ =  backTreeZ.set(1, deltaRoot);

deltaW = zeros(d,d);
backTreeW = backTreeW.set(1, deltaW);

deltaV = deltaP*node{1}';
backTreeV = backTreeV.set(1, deltaV);

topL = childL{2};
topR = childR{2};
bottom = topL+topR;
deltaUL = dLdGammaL*node{1}';
deltaUR = dLdGammaR*node{1}';
deltaU = cell(2,1);
deltaU{1}=deltaUR;
deltaU{2}=deltaUL;

backTreeU = backTreeU.set(1, deltaU);

%set delta values for all autoencoder nodes
node = sentTree.get(idx);
deltak = backTreeZ.get(sentTree.getparent(idx));
deltaW = deltak*node{1};
children = getchildren(idx);
childR = sentTree.get(max(children));
childL = sentTree(min(children));
tr = childR{1};
tl = childL{1};
zl = node{6};
zr = node{7};
a = node{4};
p = {3};
%Ul = U(1:d,:);
%Ur = U(d+1:2*d,:);
%Wk = ??

deltaZ = deltaOutput( t, p, tl, zl, tr, zr, deltak, Wk, Ul, Ur, a);
backTreeZ = backTreeZ.set(idx, deltaZ);

%set delta values for all inner nodes
%set delta values for all leaf nodes


end

