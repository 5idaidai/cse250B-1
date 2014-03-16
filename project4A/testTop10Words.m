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

totalsize=length(allSNum);
ordering = randperm(totalsize);
numSamplesTrain = ceil(totalsize*.005);
dataTrain = allSNum(ordering(1:numSamplesTrain));
labelsTrain = labels(ordering(1:numSamplesTrain));

lastidx=numSamplesTrain;
numSamplesTest = ceil(totalsize*.01);
dataTest = allSNum(ordering(lastidx:lastidx+numSamplesTest));
labelsTest = labels(ordering(lastidx:lastidx+numSamplesTest));

%BOW or NN?
method='BOW';
%method='NN';

%hyperparameters
trainInput = 0;%don't train input for now
d = 20;
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alpha = 0.2;
maxIter = 70;

if strcmp(method,'BOW')==1
    [pred, ~, ~, V, meanings] = trainBOW( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter );
    [ predTest, ~ ] = testBOW( dataTest, labelsTest, d, alpha, V, meanings );
else
    [pred, totalTime, epochTimes, totalCosts, W, U, V, meanings ] = trainNN( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter, trainInput );
    [ predTest, totalTimeTest, totalCostTest ] = testNN( words, dataTest, labelsTest, d, lambda, alpha, maxIter, W, U, V, meanings );
end

%Accuracy
[prec_train, recall_train, acc_train, f1_train] = getAccuracy(pred, labelsTrain);
[prec_test, recall_test, acc_test, f1_test] = getAccuracy(predTest, labelsTest);

fprintf('Training Accuracy: %.3f\n',acc_train);
fprintf('Testing Accuracy: %.3f\n',acc_test);

[ posidxs, poswords, negidxs, negwords] = top10Words(words, meanings, V);

disp('Top 10 Pos Words');
poswords
disp('Top 10 Neg Words');
negwords