addpath(genpath('tree/'));
addpath(genpath('codeDataMoviesEMNLP/code'));

%load test data with limit number of short sentences, same vocab
if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

totalsize=length(allSNum);
ordering = randperm(totalsize);
numSamplesTest = ceil(totalsize*.1);
dataTest = allSNum(ordering(1:numSamplesTest));
labelsTest = labels(ordering(1:numSamplesTest));

lastidx=numSamplesTest+1;
dataTrain = allSNum(ordering(lastidx:end));
labelsTrain = labels(ordering(lastidx:end));

%hyperparameters
d = 20;
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alphas = (0:0.1:1);
maxIter = 70;

accs = zeros(length(alphas),6);
bowRes = zeros(length(alphas),8);
nnRes = zeros(length(alphas),8);
nnWsRes = zeros(length(alphas),8);

aidx=1;
for alpha=alphas
    %run without training input
    [pred, ~, ~, ~, W, U, V, meanings] = trainNN( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter, 0 );
    [ predTest, ~, ~ ] = testNN( words, dataTest, labelsTest, d, lambda, alpha, maxIter, W, U, V, meanings );
    
    %Accuracy
    nnRes(aidx,1:4) = getAccuracy(pred, labelsTrain);
    nnRes(aidx,5:8) = getAccuracy(predTest, labelsTest);
    fprintf('NN Training Accuracy: %.3f\n',nnRes(aidx,3));
    fprintf('NN Testing Accuracy: %.3f\n',nnRes(aidx,3+4));
    accs(aidx,1)=nnRes(aidx,3);
    accs(aidx,2)=nnRes(aidx,3+4);
    
    %run with training input
    [predWM, ~, ~, ~, W, U, V, meanings] = trainNN( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter, 1 );
    [ predTestWM, ~, ~ ] = testNN( words, dataTest, labelsTest, d, lambda, alpha, maxIter, W, U, V, meanings );
    
    %Accuracy
    nnWsRes(aidx,1:4) = getAccuracy(predWM, labelsTrain);
    nnWsRes(aidx,5:8) = getAccuracy(predTestWM, labelsTest);
    fprintf('NN w/ Training Accuracy: %.3f\n',nnWsRes(aidx,3));
    fprintf('NN w/ Testing Accuracy: %.3f\n',nnWsRes(aidx,3+4));
    accs(aidx,3)=nnWsRes(aidx,3);
    accs(aidx,4)=nnWsRes(aidx,3+4);
        
    [predBOW, ~, ~, V, meanings] = trainBOW( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter );
    [ predTestBOW, ~ ] = testBOW( dataTest, labelsTest, d, alpha, V, meanings );
    
    %Accuracy
    bowRes(aidx,1:4) = getAccuracy(predBOW, labelsTrain);
    bowRes(aidx,5:8) = getAccuracy(predTestBOW, labelsTest);
    fprintf('BOW Training Accuracy: %.3f\n',bowRes(aidx,3));
    fprintf('BOW Testing Accuracy: %.3f\n',bowRes(aidx,3+4));
    accs(aidx,5)=bowRes(aidx,3);
    accs(aidx,6)=bowRes(aidx,3+4);
    aidx = aidx+1;
end

filename=sprintf('alphaSearchResults.mat');
save(filename);

figure;
title('Accuracy vs Alpha value');
plot(accs);