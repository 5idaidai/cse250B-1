function [ dV,dW,dU,backTreeZ, backTreeV, backTreeW, backTreeU ] =...
    backProp( sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, d, V, trainInput )
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

backTreeZ = tree(sentTree, zeros(size(d,1)));
backTreeV = tree(sentTree, zeros(size(V)));
backTreeW = tree(sentTree, zeros(size(W)));
backTreeU = tree(sentTree, zeros(size(U)));

%set delta values for root
node = sentTree.get(1);
children = sentTree.getchildren(1);
childR = sentTree.get(max(children));
childL = sentTree.get(min(children));
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

[ deltaRt, deltaP, dLdGammaL, dLdGammaR ] = deltaRoot( tl, zl, tr, zr, Ul, Ur, a, t, p, V, topL, topR, bottom);
backTreeZ =  backTreeZ.set(1, deltaRt);

deltaW = deltaRt * [childR{1};childL{1};1]';
backTreeW = backTreeW.set(1, deltaW);

deltaV = deltaP*node{1}';
backTreeV = backTreeV.set(1, deltaV);

deltaU = buildDeltaU(node{1},dLdGammaL,dLdGammaR);
backTreeU = backTreeU.set(1, deltaU);

%set delta values for all autoencoder nodes
%for i=1:length(outputItr)
    %idx = outputItr(i);
for idx=outputItr    

    node = sentTree.get(idx);
    deltak = backTreeZ.get(sentTree.getparent(idx));
    children = sentTree.getchildren(idx);
    childR = sentTree.get(max(children));
    childL = sentTree.get(min(children));
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

    [ delta, deltaP, dLdGammaL, dLdGammaR ] = deltaOutput( t, p, tl, zl, tr, zr, deltak, Wk, Ul, Ur, a, V, topL, topR, bottom);
    backTreeZ = backTreeZ.set(idx, delta);
    
    deltaW = deltaRt * [childR{1};childL{1};1]';
    backTreeW = backTreeW.set(idx, deltaW);

    deltaV = deltaP*node{1}';
    backTreeV = backTreeV.set(idx, deltaV);

    deltaU = buildDeltaU(node{1},dLdGammaL,dLdGammaR);
    backTreeU = backTreeU.set(idx, deltaU);
    
    
end

%set delta values for all inner nodes
%for i=length(innerItr)    
    %idx = innerItr(i);
for idx=innerItr    

    node = sentTree.get(idx);
    paridx = sentTree.getparent(idx);
    deltak = backTreeZ.get(paridx);
    %children = sentTree.getchildren(idx);
    %childR = sentTree.get(max(children));
    %childL = sentTree.get(min(children));
    %tr = childR{1};
    %tl = childL{1};
    %zl = node{6};
    %zr = node{7};
    a = node{4};
    p = node{3};
    %Ul = U(1:d,:);
    %Ur = U(d+1:2*d,:);
    
    parent = sentTree.getparent(idx);
    nodes = sentTree.getchildren(parent);
    nodeL = min(nodes);
    if nodeL==idx
        Wk = W(:,1:d);
    else
        Wk = W(:,d+1:2*d);
    end

    [ delta, deltaP ] = deltaNonOutput( a, deltak, Wk, V, t, p );
    backTreeZ = backTreeZ.set(idx, delta);
    
    deltaW = deltaRt * [childR{1};childL{1};1]';
    backTreeW = backTreeW.set(idx, deltaW);

    deltaV = deltaP*node{1}';
    backTreeV = backTreeV.set(idx, deltaV);

    %topL = childL{2};
    %topR = childR{2};
    %bottom = topL+topR;
    %deltaUL = dLdGammaL*node{1}';
    %deltaUR = dLdGammaR*node{1}';
    %deltaU = cell(2,1);
    %deltaU{1}=deltaUR;
    %deltaU{2}=deltaUL;

    %backTreeU = backTreeU.set(idx, deltaU);
    
    
end

if trainInput
    %set delta values for all leaf nodes
    %for i=length(inputItr)    
    %    idx = inputItr(i);
    for idx=inputItr

        node = sentTree.get(idx);
        deltak = backTreeZ.get(sentTree.getparent(idx));
        %children = sentTree.getchildren(idx);
        %childR = sentTree.get(max(children));
        %childL = sentTree.get(min(children));
        %tr = childR{1};
        %tl = childL{1};
        %zl = node{6};
        %zr = node{7};
        a = node{4};
        %p = node{3};
        %Ul = U(1:d,:);
        %Ur = U(d+1:2*d,:);

        parent = sentTree.getparent(idx);
        nodes = sentTree.getchildren(parent);
        nodeL = min(nodes);
        if nodeL==idx
            Wk = W(:,1:d);
        else
            Wk = W(:,d+1:2*d);
        end

        [ delta ] = deltaInput( a, deltak, Wk, trainInput);
        backTreeZ = backTreeZ.set(idx, delta);

        %deltaW = deltak*[node{1};1]';
        %backTreeW = backTreeW.set(idx, deltaW);

        %deltaV = deltaP*node{1}';
        %backTreeV = backTreeV.set(idx, deltaV);

        %topL = childL{2};
        %topR = childR{2};
        %bottom = topL+topR;
        %deltaUL = dLdGammaL*node{1}';
        %deltaUR = dLdGammaR*node{1}';
        %deltaU = cell(2,1);
        %deltaU{1}=deltaUR;
        %deltaU{2}=deltaUL;

        %backTreeU = backTreeU.set(idx, deltaU);
    end
end

%display the deriv trees (and sizes of matrices)
if 0
    disp(backTreeZ.tostring());
    disp(backTreeV.tostring());
    disp(backTreeW.tostring());
    disp(backTreeU.tostring());
end

iterator = sentTree.breadthfirstiterator;
dV = zeros(size(V));
dW = zeros(size(W));
dU = zeros(size(U));

for i=iterator
    if ~sentTree.isleaf(i)
        dV = dV + backTreeV.get(i);
        dW = dW + backTreeW.get(i);
        dU = dU + backTreeU.get(i);
    end
end

end

