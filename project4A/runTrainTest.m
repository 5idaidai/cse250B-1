addpath(genpath('tree/'));

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
lambda = 0.001/length(allSNum);
alpha = 0.4;

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

%iterate through all sentences
for i=1:length(allSNum)
    %get sentence
    sent=allSNum(i);
    sent=sent{1,1};
    numWords=length(sent);
    
    %build up sentence binary tree, and perform feed forward
    %   algorithm at the same time
    [sentTree, outputItr, innerItr, inputItr] = buildTree(sent, meanings, numWords, W, U, V, d);
    
    %disp(sentTree.tostring());
    %sentTree;
    %pause;
    
    %backpropagate
    %t=rand(1);
    t=labels(i);
    t=[t; 1-t];
    [dV,dW,dU,backTreeZ, backTreeV, backTreeW, backTreeU] =...
        backProp(sentTree, t, outputItr, innerItr, inputItr, U, W, d, V, trainInput);
    
    pause;
    
    %SGD update
    V = V - labmda*dV - (labmda/2)*(V.^2);
    W = W - labmda*dW - (labmda/2)*(W.^2);
    U = U - labmda*dU - (labmda/2)*(U.^2);
end

