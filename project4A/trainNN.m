function [ pred, totalTime, epochTimes ] = trainNN( words, allSNum, labels, d, lambda, alpha, maxIter, trainInput )
%trainNN build and train the neural network

%init meaning vectors for each word to random values
meanings = normrnd(0,1,d,size(words,2));
numExamples=length(allSNum);

%init W and b randomly
W = rand(d,2*d);
b = zeros(d,1);
W = [W,b];

%init U and c for backpropagation
U = rand(2*d,d);
c = zeros(2*d,1);
U = [U,c];

%init V for prediction
V = rand(2,d);

rootPreds = zeros(2,numExamples);

%Training
totTic=tic;
epochTimes=zeros(maxIter,1);

for epoch=1:maxIter
    if epoch==1 || mod(epoch,maxIter/5)==0
        disp(epoch);    
    end
    eTic = tic;
    
    %iterate through all sentences
    for i=1:numExamples
        %get sentence
        sent=allSNum(i);
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
        [sentTree, outputItr, innerItr, inputItr] =...
            buildTree(sent, meanings, numWords, W, U, V, d, t, alpha, trainInput);

        %store root node prediction: for predicting sentence meaning
        root = sentTree.get(1);
        rootPreds(:,i) = root{3};
        
        %backpropagate
        if trainInput
            [dW,dU,dV,~,dMeaning] =...
                backProp(sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, V, d, alpha, trainInput);
        else
            [dW,dU,dV] = backProp(sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, V, d, alpha, trainInput);
        end
        
        %Regularized SGD update
        newW = W + lambda(1)*dW;
        newU = U + lambda(2)*dU;
        V = V + lambda(3)*dV;
        
        if trainInput
            meanings = meanings + lambda(4)*dMeaning;
        end

        %Don't regularize intercept
        W = [newW(:,1:end-1),W(:,end)];
        U = [newU(:,1:end-1),U(:,end)];
    end
    
    epochTimes(epoch) = toc(eTic);    
end

totalTime = toc(totTic);
fprintf('SGD_NN took %f seconds (aka %f minutes).\n\n',totalTime,totalTime/60);
plot(epochTimes);

pred = zeros(numExamples,1);
rootPreds(2,:)=1-rootPreds(1,:);
for i=1:numExamples
     pred(i)=find(rootPreds(:,i)>0.5)-1;
end

end