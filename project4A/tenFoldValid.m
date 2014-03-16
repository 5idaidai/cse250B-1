addpath(genpath('tree/'));
addpath(genpath('codeDataMoviesEMNLP/code'));

if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

%hyperparameters
d = 20;
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alphas = 0.2;
maxIter = 70;
numfolds = 10;

Labels = num2cell(labels');
AllSNum = [allSNum,Labels];

totalsize=length(allSNum);
ordering = randperm(totalsize);
numInFold=floor(totalsize/numfolds);

startidx=1;
stopidx=numInFold;
numTestSamples = floor(numInFold * .1);

accs = zeros(numfolds,6);
bowRes = zeros(numfolds,8);
nnRes = zeros(numfolds,8);
nnWsRes = zeros(numfolds,8);

for i=1:10
    teststop = startidx+numTestSamples;
    testidxs=ordering(startidx:teststop);    
    testing = AllSNum(testidxs,:);
    dataTest = testing(:,1);
    labelsTest = cell2mat(testing(:,2));

    trainidxs=ordering(teststop+1:stopidx);
    training = AllSNum(trainidxs,:);
    dataTrain = training(:,1);
    labelsTrain = cell2mat(training(:,2));
    
    %run without training input
    [pred, ~, ~, ~, W, U, V, meanings] = trainNN( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter, 0 );
    [ predTest, ~, ~ ] = testNN( words, dataTest, labelsTest, d, lambda, alpha, maxIter, W, U, V, meanings );
    
    %Accuracy
    nnRes(i,1:4) = getAccuracy(pred, labelsTrain);
    nnRes(i,5:8) = getAccuracy(predTest, labelsTest);
    fprintf('NN Training Accuracy: %.3f\n',nnRes(i,3));
    fprintf('NN Testing Accuracy: %.3f\n',nnRes(i,3+4));
    accs(i,1)=nnRes(i,3);
    accs(i,2)=nnRes(i,3+4);
    
    %run with training input
    [predWM, ~, ~, ~, W, U, V, meanings] = trainNN( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter, 1 );
    [ predTestWM, ~, ~ ] = testNN( words, dataTest, labelsTest, d, lambda, alpha, maxIter, W, U, V, meanings );
    
    %Accuracy
    nnWsRes(i,1:4) = getAccuracy(predWM, labelsTrain);
    nnWsRes(i,5:8) = getAccuracy(predTestWM, labelsTest);
    fprintf('NN w/ Training Accuracy: %.3f\n',nnWsRes(i,3));
    fprintf('NN w/ Testing Accuracy: %.3f\n',nnWsRes(i,3+4));
    accs(i,3)=nnWsRes(i,3);
    accs(i,4)=nnWsRes(i,3+4);
        
    [predBOW, ~, ~, V, meanings] = trainBOW( words, dataTrain, labelsTrain, d, lambda, alpha, maxIter );
    [ predTestBOW, ~ ] = testBOW( dataTest, labelsTest, d, alpha, V, meanings );
    
    %Accuracy
    bowRes(i,1:4) = getAccuracy(predBOW, labelsTrain);
    bowRes(i,5:8) = getAccuracy(predTestBOW, labelsTest);
    fprintf('BOW Training Accuracy: %.3f\n',bowRes(i,3));
    fprintf('BOW Testing Accuracy: %.3f\n',bowRes(i,3+4));
    accs(i,5)=bowRes(i,3);
    accs(i,6)=bowRes(i,3+4);
   
    filename=sprintf('tenFoldData-%.0f.mat',i);
    save(filename);
    startidx=stopidx+1;
    stopidx=stopidx+numInFold;
    
end

filename=sprintf('tenFoldValidResults.mat');
save(filename);

