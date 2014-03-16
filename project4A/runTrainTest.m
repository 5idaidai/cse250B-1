addpath(genpath('tree/'));
addpath(genpath('codeDataMoviesEMNLP/code'));

%load test data with limit number of short sentences, same vocab
% load('testdata.mat');
if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

numSamplesTrain = ceil(length(allSNum)*.002);
[dataTrain,idx] = datasample(allSNum,numSamplesTrain,'Replace',false);
labelsTrain = labels(idx);

numSamplesTest = ceil(length(allSNum)*.002);
[dataTest,idxTest] = datasample(allSNum,numSamplesTest,'Replace',false);
labelsTest = labels(idxTest);

% data = allSNum;
% sampledLabels = labels;

%BOW or NN?
%method='BOW';
method='NN';

%hyperparameters
trainInput = 0;%don't train input for now
d = 20;
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alpha = 0.2;
maxIter = 70;

if strcmp(method,'BOW')==1
    [pred] = trainBOW( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter );
else
    [pred, totalTime, epochTimes, totalCosts, W, U, V, meanings] = trainNN( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter, trainInput );
    [ predTest, totalTimeTest, totalCostTest ] = testNN( words, dataTest, labelsTest, d, lambda, alpha, maxIter, W, U, V, meanings );
end

%Accuracy
[prec_train, recall_train, acc_train, f1_train] = getAccuracy(pred, labelsTrain);
[prec_test, recall_test, acc_test, f1_test] = getAccuracy(predTest, labelsTest);

fprintf('Training Accuracy: %.3f\n',acc_train);
fprintf('Testing Accuracy: %.3f\n',acc_test);