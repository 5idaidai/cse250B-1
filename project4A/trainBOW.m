function [ pred, totalTime, epochTimes, V, meanings ] = trainBOW( words, data, labels, d, lambda, alpha, maxIter )
%trainNN build and train the neural network

%init meaning vectors for each word to random values
meanings = normrnd(0,1,d,size(words,2));
numExamples=length(data);

%init V for prediction
V = -1 + (1+1)*rand(1,d);

runPreds = zeros(1,numExamples);
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
        sent=data(i);
        sent=sent{1,1};
        numWords=length(sent);
        
        %skip sentences of less than 2 words because our our neural nets
        %are defined for these
        if numWords<2
            continue;
        end
        
        t=labels(i);

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
        runPreds(i)=p;
        
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
% plot(epochTimes);

predd = runPreds>0.5;
pred = zeros(size(predd));
pred = pred + predd;

end

