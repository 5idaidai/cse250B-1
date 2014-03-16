function [ pred, totalTime, epochTimes ] = trainBOW( words, allSNum, labels, d, lambda, alpha, maxIter )
%trainNN build and train the neural network

%init meaning vectors for each word to random values
meanings = normrnd(0,1,d,size(words,2));
numExamples=length(allSNum);

%init W and b randomly
%a + (b-a).*rand(100,1)
W = -1 + (1+1)*rand(d,2*d);
b = zeros(d,1);
W = [W,b];

%init U and c for backpropagation
U = -1 + (1+1)*rand(2*d,d);
c = zeros(2*d,1);
U = [U,c];

%init V for prediction
V = -1 + (1+1)*rand(2,d);

runPreds = zeros(2,numExamples);
dM = zeros(size(meanings));

%Training
totTic=tic;
epochTimes=zeros(maxIter,1);

for epoch=1:maxIter
    if epoch==1 || mod(epoch,maxIter/5)==0
        disp(epoch);    
    end
    eTic = tic;
    
    %iterate through all sentences
    for i=1:numExamples
        %get sentence
        sent=allSNum(i);
        sent=sent{1,1};
        numWords=length(sent);
        
        %skip sentences of less than 2 words because our our neural nets
        %are defined for these
        if numWords<2
            continue;
        end
        
        tl=labels(i);
        t=[tl; 1-tl];

        %Run Bag of Words predictions on each sentence
        [ BOWlist ] = bagOfWords( sent, meanings, numWords, V, t, alpha );
        
        %backpropagate
        [ dV, BOWlist, dMeaning ] = backPropBOW( BOWlist, V, alpha, t, meanings, numWords);
        
        %Get average word meaning and predict sentence meaning
        wordMeaning=zeros(d,numWords);
        for w=1:numWords
            word=BOWlist{w,1};
            wordMeaning(:,w)=word{1};
        end
        aveMeaning=mean(wordMeaning,2);
        p = predictNode(aveMeaning,V);
        runPreds(:,i)=p;
        
        %Regularized SGD update
        V = V + lambda(3)*dV;
        
        %Train meanings
        dM = dM + dMeaning;
        
    end
    %Update meanings
    meanings = meanings + lambda(4)*dM;
    
    epochTimes(epoch) = toc(eTic);    
end

totalTime = toc(totTic);
fprintf('SGD_BOW took %f seconds (aka %f minutes).\n\n',totalTime,totalTime/60);
plot(epochTimes);

pred = zeros(numExamples,1);
runPreds(2,:)=1-runPreds(1,:);
for i=1:numExamples
     pred(i)=find(runPreds(:,i)>0.5)-1;
end

end

