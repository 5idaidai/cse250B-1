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
V = rand(1,d);

totTic=tic;


i=3;%randi(length(allSNum),1);
sent=allSNum(i);
sent=sent{1,1};
numWords=length(sent);

%skip sentences of less than 2 words because our our neural nets
%are defined for these
if numWords>=2
    t=labels(i);

    %build up sentence binary tree, and perform feed forward
    %algorithm at the same time
    [sentTree, outputItr, innerItr, inputItr] = buildTree(sent, meanings, numWords, W, U, V, d, t, alpha, trainInput, 1);

    %backpropagate
    [ dW ] = backProp( sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, V, d, alpha, trainInput );
    %dW = dW * lambda(1);
    

    %Numerical Differentiaton
    A = (-1:.5:2);
    E = 10.^A;
    E = sort(E);
    Diff=zeros(length(E),1);
    for i=1:length(E)
        e=E(i);
        [ numDiffW ] = numDiff( outputItr, innerItr, sentTree, W, U, V, d, t, alpha, e, lambda );
        %numDiffW = numDiffW * lambda(1);
        %Check derivatives
        D = sum(sum((numDiffW(1:end-1) - dW(1:end-1)).^2));
        Diff(i)=D;
    end
else
    i=i+1;
end

totalTime = toc(totTic);
fprintf('Checking one tree took %f seconds (aka %f minutes).\n\n',totalTime,totalTime/60);
fprintf('The Euclidean distance is %f\n', D);
