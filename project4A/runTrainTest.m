
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
W = rand(d,2*d);
b = rand(d,1);

%init U and c for backpropagation
U = rand(2*d,d);
c = rand(2*d,1);

%init V for prediction
V = rand(1,d);

%iterate through all sentences
for i=1:length(allSNum)
    %get sentence
    sent=allSNum(i);
    sent=sent{1,1};
    numWords=length(sent);
    
    %get meaning vectors for each word
    sentMean = meanings(:,sent);
    
    sentTree = buildTree(sentMean, numWords, W, b, U, c, V, d);
 
    iterator = sentTree.breadthfirstiterator;
    
    for i = iterator
        %only iterate through non-leaf nodes for backprop
        check = sentTree.isleaf(i);
        if check == true
        end
    end
    
    disp(sentTree.tostring());
    sentTree;
    pause;
end