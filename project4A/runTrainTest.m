
if exist('words','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');
end
if exist('allSNum','var') == 0
    load('codeDataMoviesEMNLP/data/rt-polaritydata/RTData_CV1.mat','allSNum','labels');
end

%init meaning vectors for each word to random values
d = 20;
meanings = normrnd(0,1,d,size(words,2));

%init W and b randomly
W = rand(d,2*d+1);
%b = rand(d,1);
%W = [W,b];

%init U and c for backpropagation
U = rand(2*d,d+1);
%c = rand(2*d,1);
%U = [U,c];

%init V for prediction
V = rand(2,d);

%iterate through all sentences
for i=1:length(allSNum)
    %get sentence
    sent=allSNum(i);
    sent=sent{1,1};
    numWords=length(sent);
    
    %get meaning vectors for each word
    sentMean = meanings(:,sent);
    
    %build up sentence binary tree, and perform feed forward
    %   algorithm at the same time
    [sentTree, outputItr, innerItr, inputItr] = buildTree(sentMean, numWords, W, U, V, d);
    
    disp(sentTree.tostring());
    sentTree;
    pause;
    
    %backpropagate
    %t=rand(1);
    t=labels(i);
    t=[t; 1-t];
    [backTreeZ, backTreeV, backTreeW, backTreeU] = backProp(sentTree, t, outputItr, innerItr, inputItr, U, W, d, V);
    
    disp(backTreeZ.tostring());
    disp(backTreeV.tostring());
    disp(backTreeW.tostring());
    disp(backTreeU.tostring());
    pause;
end

