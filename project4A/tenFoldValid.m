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

Labels = num2cell(labels');
AllSNum = [allSNum,Labels];

totalsize=length(allSNum);
ordering = randperm(totalsize);
numInFold=floor(totalsize/10);

startidx=1;
stopidx=numInFold;

accs = zeros(length(alphas),6);
bowRes = zeros(length(alphas),8);
nnRes = zeros(length(alphas),8);
nnWsRes = zeros(length(alphas),8);

for i=1:10
    testidxs=ordering(startidx:stopidx);
    testing = AllSNum(testidxs,:);
    dataTest = testing(:,1);
    labelsTest = cell2mat(testing(:,2));

    trainidxs=~ismember(ordering,testidxs);
    trainidxs=ordering(trainidxs);
    training = AllSNum(trainidxs,:);
    dataTraining = training(:,1);
    labelsTraining = cell2mat(training(:,2));
    
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
   
    filename=sprintf('tenFoldData-%.0f.mat',i);
    save(filename);
    startidx=stopidx+1;
    stopidx=stopidx+numInFold;
    
end

filename=sprintf('tenFoldValidResults.mat');
save(filename);

