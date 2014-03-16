function [ pred, totalTime, totalCost ] = testNN( words, data, labels, d, lambda, alpha, maxIter, W, U, V, meanings )
%trainNN build and train the neural network

numExamples=length(data);

rootPreds = zeros(2,numExamples);

%Training
totTic=tic;
    
totalCost = 0;
%iterate through all sentences
for i=1:numExamples
    %get sentence
    sent=data(i);
    sent=sent{1,1};
    numWords=length(sent);

    %skip sentences of less than 2 words because our our neural nets
    %are defined for these
    if numWords<2
        continue;
    end

    tl=labels(i);
    t=[tl; 1-tl];

    %build up sentence binary tree, and perform feed forward
    %   algorithm at the same time
    [sentTree, ~, ~, ~, sentCost] =...
        buildTree(sent, meanings, numWords, W, U, V, d, t, alpha, 0);

    %store root node prediction: for predicting sentence meaning
    root = sentTree.get(1);
    rootPreds(:,i) = root{3};

    totalCost = totalCost + sentCost;
end

totalCost = totalCost + (lambda(1:3)./2)*([norm(W), norm(U), norm(V)].^2)';

totalTime = toc(totTic);
fprintf('SGD_NN took %f seconds (aka %f minutes) to test.\nTotal cost: %f\n',totalTime,totalTime/60,totalCost);

pred = zeros(numExamples,1);
rootPreds(2,:)=1-rootPreds(1,:);
for i=1:numExamples
     pred(i)=find(rootPreds(:,i)>0.5)-1;
end

end