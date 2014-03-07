
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
    numNodes = numWords;
    %columns: child 1, child 2, meaning vector
    %0 in child column indicates no child
    nodes = cell(size(sentMean,2),1);
    for i=1:numWords;
        nodes{i} = tree(sentMean(:,i));
    end
    
    while(numNodes>1)
        %inner loop of tree construction: 
        %for each pair of nodes (starting with leaf nodes i.e. the sentence)
        err = ones(numNodes-1,3);
        k = 1;
        numCreated = 0;
        for j=1:numNodes-1
            %compute RAE error
            node1=j;
            node2=j+1;

            xi = nodes{node1}.Node{1};
            xj = nodes{node2}.Node{1};

            ni = 1;
            nj = 1;

            xk = meaningFunc(xi,xj,W,b);
            [err(j,1), zi, zj] = raeError( xk, xi, xj, ni, nj, U, c, d );
            err(j,2) = node1;
            err(j,3) = node2;
            k = k + 1;
        end    %end inner loop
        [val,idx] = min(err(:,1));
        child1 = err(idx,2);
        child2 = err(idx,3);
        xi = nodes{child1}.Node{1};
        xj = nodes{child2}.Node{1};
        xk = meaningFunc(xi,xj,W,b);
        %zk = predictNode(xk,V);

        newnode = tree(xk);
        newnode = newnode.graft(1,nodes{child1});
        newnode = newnode.graft(1,nodes{child2});

        newnodes = cell(numNodes-1,1);
        for i=1:numNodes
            if i<child1
                newnodes{i} = nodes{i};
            elseif i==child1
                newnodes{i} = newnode;
            elseif i>child2
                newnodes{i-1} = nodes{i};
            end
        end
        nodes=newnodes;

        numCreated = numCreated + 1;
        numNodes = numNodes - 1;    
    end
    
    disp(nodes{1}.tostring)
    nodes{1}
    size(nodes)
end