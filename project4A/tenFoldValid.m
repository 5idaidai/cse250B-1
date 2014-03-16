addpath(genpath('tree/'));
addpath(genpath('codeDataMoviesEMNLP/code'));

if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

Labels = num2cell(labels');
AllSNum = [allSNum,Labels];

totalsize=length(allSNum);
ordering = randperm(totalsize);
numInFold=floor(totalsize/10);

startidx=1;
stopidx=numInFold;

for i=1:10
    numSamplesTest = ceil(numInFold*.01);
    dataTest = AllSNum(ordering(startidx:numSamplesTest),:);

    lastidx=numSamplesTest;
    dataTrain = AllSNum(ordering(numSamplesTest+1:stopidx),:);
   
    filename=sprintf('tenFoldData-%.0f.mat',i);
    save(filename);
    startidx=stopidx+1;
    stopidx=stopidx+numInFold;
    
end

