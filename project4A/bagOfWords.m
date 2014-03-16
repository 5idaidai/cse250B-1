function [ BOWlist ] = bagOfWords( sent, wordMeanings, numWords, V, t, alpha )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    numCells = 6;
    %Build a list of of input nodes
    inputNodeList = cell(numWords,1);
    for i=1:numWords;
        node=cell(numCells,1);
        %each node contains the following:
        % 1: meaning vector
        % 2: z, predicted label
        % 3: a, activation
        % 4: delta vector
        % 5: word index
        % 6: E2, prediction error
                
        node{5} = sent(i);
        node{1} = wordMeanings(:,node{5});
        node{3} = node{1};%activation for input nodes is the same as their meaning vector
        inputNodeList{i} = node;
    end
    
    %Build a list of output nodes for the input nodes in a 1:1 ratio
    outputNodeList = cell(numWords,1);
    for i=1:numWords;
        node=cell(numCells,1);
        child=inputNodeList{i};
        xi = child{1};
        [pk,ak] = predictNode(xi,V);
        E2 = predError(t,pk,alpha);
        
        node{1} = xi;
        node{2} = pk;
        node{3} = ak;
        node{5} = sent(i);
        node{6} = E2;
        outputNodeList{i} = node;
    end
    
    BOWlist = [outputNodeList,inputNodeList];
end

