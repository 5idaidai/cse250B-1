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

% numSamples = ceil(length(allSNum)*.1);
% [data,idx] = datasample(allSNum,numSamples,'Replace',false);
% sampledLabels = labels(idx);
data = allSNum;
sampledLabels = labels;

%hyperparameters
trainInput = 0;%don't train input for now
d = 20;
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alpha = 0.2;
maxIter = 70;

[pred, totalTime, epochTimes, totalCosts] = trainNN( words, data, sampledLabels, d, lambda, alpha, maxIter, trainInput );

%Accuracy
%dec_val = sigmoid(W*rootPreds' + b(:,ones(numExamples,1)));
%pred = 1*(dec_val > 0.5);
[prec_train, recall_train, acc_train, f1_train] = getAccuracy(pred, sampledLabels);
fprintf('Training Accuracy: %.3f\n',acc_train);