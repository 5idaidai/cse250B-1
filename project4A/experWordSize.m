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
ds = [20,100];
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alpha = 0.2;
maxIter = 70;

accs = zeros(length(ds),2);
nnRes = zeros(length(ds),8);

didx=1;
for d=ds
    %run without training input
    [pred, ~, ~, ~, W, U, V, meanings] = trainNN( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter, 0 );
    [ predTest, ~, ~ ] = testNN( words, dataTest, labelsTest, d, lambda, alpha, maxIter, W, U, V, meanings );
    
    %Accuracy
    nnRes(didx,1:4) = getAccuracy(pred, labelsTrain);
    nnRes(didx,5:8) = getAccuracy(predTest, labelsTest);
    fprintf('NN Training Accuracy: %.3f\n',nnRes(didx,3));
    fprintf('NN Testing Accuracy: %.3f\n',nnRes(didx,3+4));
    accs(didx,1)=nnRes(didx,3);
    accs(didx,2)=nnRes(didx,3+4);
end

filename=sprintf('wordsSizeResults.mat');
save(filename);
