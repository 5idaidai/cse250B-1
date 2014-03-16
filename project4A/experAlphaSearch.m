addpath(genpath('tree/'));
addpath(genpath('codeDataMoviesEMNLP/code'));

%load test data with limit number of short sentences, same vocab
if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

%take 10% of data for this experiment
numSamples = ceil(length(allSNum)*.1);
[data,idx] = datasample(allSNum,numSamples,'Replace',false);
sampledLabels = labels(idx);

%hyperparameters
trainInput = 0;%don't train input for now
d = 20;
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alphas = [0:0.1:1];
maxIter = 70;

accs = zeros(length(alphas));

aidx=1;
for alpha=alphas
    [pred, acc] = trainNN( words, data, sampledLabels, d, lambda, alpha, maxIter, trainInput );

    %training Accuracy
    [prec_train, recall_train, acc_train, f1_train] = getAccuracy(pred, labels);
    fprintf('Training Accuracy: %.3f\n',acc_train);
    accs(aidx) = acc_train;
end

figure;
title('Accuracy vs Alpha value');
plot(accs);