addpath(genpath('tree/'));

%load test data with limit number of short sentences, same vocab
%load('testdata.mat');
if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

%hyperparameters
trainInput = 0;%don't train input for now
d = 20;
lambda = 0.001/length(allSNum);
alpha = 0.4;
maxIter = 70;

%init meaning vectors for each word to random values
meanings = normrnd(0,1,d,size(words,2));

%init W and b randomly
W = rand(d,2*d+1);
%b = rand(d,1);
%W = [W,b];

%init U and c for backpropagation
U = rand(2*d,d+1);
%c = rand(2*d,1);
%U = [U,c];

%init V for prediction
V = rand(2,d);

totTic=tic;
epochTimes=zeros(maxIter,1);

for epoch=1:maxIter
    %if epoch==1 || mod(epoch,10)==0
        disp(epoch);    
    %end
    eTic = tic;
    
    %iterate through all sentences
    for i=1:length(allSNum)
        %get sentence
        sent=allSNum(i);
        
        %skip sentences of less than 2 words because our our neural nets
        %are defined for these
        if length(sent)<2
            continue;
        end
        
        sent=sent{1,1};
        numWords=length(sent);
        t=labels(i);
        t=[t; 1-t];

        %build up sentence binary tree, and perform feed forward
        %   algorithm at the same time
        [sentTree, outputItr, innerItr, inputItr] = buildTree(sent, meanings, numWords, W, U, V, d);

        %disp(sentTree.tostring());
        %sentTree;
        %pause;

        %backpropagate
        [dV,dW,dU,backTreeZ, backTreeV, backTreeW, backTreeU] =...
            backProp(sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, d, V, trainInput);

        %Regularized SGD update
        V = V - lambda*dV - (lambda/2)*(V.^2);
        newW = W - lambda*dW - (lambda/2)*(W.^2);
        newU = U - lambda*dU - (lambda/2)*(U.^2);

        %Don't regularize intercept
        W = [newW(:,1:end-1),W(:,end)];
        U = [newU(:,1:end-1),U(:,end)];
    end
    
    epochTimes(epoch) = toc(eTic);    
end

totalTime = toc(totTic);
fprintf('SGD_NN took %f seconds (aka %f minutes).\n\n',totalTime,totalTime/60);
plot(epochTimes);
