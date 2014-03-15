if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

tempNum=cell(5,1);
tempLabels=zeros(5,1);

tempNum{1}=allSNum{3};
tempLabels(1)=labels(3);

tempNum{2}=allSNum{27};
tempLabels(2)=labels(27);

tempNum{3}=allSNum{84};
tempLabels(3)=labels(84);

tempNum{4}=allSNum{6054};
tempLabels(4)=labels(6054);

tempNum{5}=allSNum{6071};
tempLabels(5)=labels(6071);

allSNum=tempNum;
labels=tempLabels;

save('testdata.mat','allSNum','labels','words');