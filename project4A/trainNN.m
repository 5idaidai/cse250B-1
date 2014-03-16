function [ pred, totalTime, epochTimes, totalCosts, W, U, V, meanings ] = trainNN( words, allSNum, labels, d, lambda, alpha, maxIter, trainInput )
%trainNN build and train the neural network

%init meaning vectors for each word to random values
meanings = normrnd(0,1,d,size(words,2));
numExamples=length(allSNum);

%init W and b randomly
%a + (b-a).*rand(100,1)
W = -1 + (1+1)*rand(d,2*d);
b = zeros(d,1);
W = [W,b];

%init U and c for backpropagation
U = -1 + (1+1)*rand(2*d,d);
c = zeros(2*d,1);
U = [U,c];

%init V for prediction
V = -1 + (1+1)*rand(2,d);

rootPreds = zeros(2,numExamples);
if trainInput
    dM = sparse(size(meanings,1),size(meanings,2));
end

%Training
totTic=tic;
epochTimes=zeros(maxIter,1);
oldCost = 1;
totalCosts = zeros(maxIter,1);

for epoch=1:maxIter
    if epoch==1 || mod(epoch,maxIter/5)==0
        disp(epoch);    
    end
        
    eTic = tic;
    
    totalCost = 0;
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
        [sentTree, outputItr, innerItr, inputItr, sentCost] =...
            buildTree(sent, meanings, numWords, W, U, V, d, t, alpha, trainInput, 1);
        %sentCost = totalError(outputItr,innerItr,alpha,sentTree);
        
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
            dM = dM + dMeaning;
        end

        %Don't regularize intercept
        W = [newW(:,1:end-1),W(:,end)];
        U = [newU(:,1:end-1),U(:,end)];
        
        totalCost = totalCost + sentCost;
    end

    if trainInput
        meanings = meanings + lambda(4)*dM;
    end
        
    totalCost = totalCost + (lambda(1:3)./2)*([norm(W), norm(U), norm(V)].^2)';
    totalCosts(epoch) = totalCost;
    
    epochTimes(epoch) = toc(eTic);
        
    if abs(totalCost - oldCost) <= 1e-8
        fprintf('Converged in %d epochs\n',epoch);
        break;
    end
    oldCost = totalCost;
end

totalTime = toc(totTic);
fprintf('SGD_NN took %f seconds (aka %f minutes).\n\n',totalTime,totalTime/60);
%plot(epochTimes);

% figure;
% title('Total Cost per Epoch');
% plot(totalCosts);

pred = zeros(numExamples,1);
rootPreds(2,:)=1-rootPreds(1,:);
for i=1:numExamples
     pred(i)=find(rootPreds(:,i)>0.5)-1;
end

end