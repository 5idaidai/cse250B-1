addpath(genpath('tree/'));
addpath(genpath('codeDataMoviesEMNLP/code'));

%load test data with limit number of short sentences, same vocab
load('testdata.mat');
if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

%hyperparameters
trainInput = 0;%don't train input for now
d = 20;
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alpha = 0.2;
maxIter = 70;

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
        
        t=labels(i);
        t=[t; 1-t];

        %build up sentence binary tree, and perform feed forward
        %   algorithm at the same time
        [sentTree, outputItr, innerItr, inputItr] =...
            buildTree(sent, meanings, numWords, W, U, V, d, t, alpha, trainInput);

        %store root node prediction: for predicting sentence meaning
        root = sentTree.get(1);
        rootPreds(:,i) = root{3};
        
        %backpropagate
        if trainInput
            [dW,dU,dV,backTreeZ,dMeaning] =...
                backProp(sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, V, d, alpha, trainInput);
        else
            [dW,dU,dV] = backProp(sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, V, d, alpha, trainInput);
        end
        
        %Regularized SGD update
        newW = W - lambda(1)*dW;
        newU = U - lambda(2)*dU;
        V = V - lambda(3)*dV;
        
        if trainInput
            meanings = meanings - lambda(4)*dMeaning;
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

%Accuracy
rootPreds(2,:)=1-rootPreds(1,:);
for i=1:numExamples
     pred(i)=find(rootPreds(:,1)>0.5)-1;
end

%dec_val = sigmoid(W*rootPreds' + b(:,ones(numExamples,1)));
%pred = 1*(dec_val > 0.5);
gold = labels;
[prec_train, recall_train, acc_train, f1_train] = getAccuracy(pred, gold);

