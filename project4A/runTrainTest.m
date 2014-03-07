
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
V = rand(2,d);

%iterate through all sentences
for i=1:length(allSNum)
    %get sentence
    sent=allSNum(i);
    sent=sent{1,1};
    numWords=length(sent);
    
    %get meaning vectors for each word
    sentMean = meanings(:,sent);
    
    %greedy tree algorithm
    numNodes = numWords - 1;
    %columns: child 1, child 2, meaning vector
    %0 in child column indicates no child
    tree = cell(numNodes,3);
    
    %inner loop of tree construction: 
    %for each pair of nodes (starting with leaf nodes i.e. the sentence)
    numNodes = numWords - 1;
    err = ones(numNodes,3);
    k = 1;
    for j=1:numNodes
        %compute RAE error
        node1=j;
        node2=j+1;
        
        xi = meanings(:,sent(node1));
        xj = meanings(:,sent(node2));
        
        ni = 1;
        nj = 1;
        
        err(j,1) = raeError( k, xi, xj, ni, nj, W, b, U, c, d );
        err(j,2) = node1;
        err(j,3) = node2;
        k = k + 1;
    end
    [val,idx] = min(err(:,1));
    child1 = err(idx,2);
    child2 = err(idx,3);
    numNodes = numNodes - 1;
    %end inner loop
end