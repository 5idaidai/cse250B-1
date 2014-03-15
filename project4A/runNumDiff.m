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
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alpha = 0.4;
maxIter = 1;

%init meaning vectors for each word to random values
meanings = normrnd(0,1,d,size(words,2));

%init W and b randomly
W = rand(d,2*d);
b = zeros(d,1);
W = [W,b];

%init U and c for backpropagation
U = rand(2*d,d);
c = zeros(2*d,1);
U = [U,c];

%init V for prediction
v1 = rand(1,d);
v2 = 1-v1;
V = zeros(2,d);
V(1,:)=v1;
V(2,:)=v2;
%V = rand(1,d);

totTic=tic;


i=randi(length(allSNum),1);
sent=allSNum(i);
sent=sent{1,1};
numWords=length(sent);

%skip sentences of less than 2 words because our our neural nets
%are defined for these
if numWords>=2
    t=labels(i);
    t=[t; 1-t];

    %build up sentence binary tree, and perform feed forward
    %algorithm at the same time
    [sentTree, outputItr, innerItr, inputItr] = buildTree(sent, meanings, numWords, W, U, V, d, t, alpha, trainInput);

    %backpropagate
    [ dW,dU,dV, backTreeZ ] = backProp( sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, V, d, alpha, trainInput );

    %Numerical Differentiaton
    E=1e-6;
    [ numDiffW ] = numDiff( outputItr, innerItr, sentTree, W, U, V, d, t, alpha, E, lambda );

    %Check derivatives
    D = sum(sum((numDiffW - dW).^2));
else
    i=i+1;
end

totalTime = toc(totTic);
fprintf('Checking one tree took %f seconds (aka %f minutes).\n\n',totalTime,totalTime/60);
fprintf('The Euclidean distance is %f', D);
