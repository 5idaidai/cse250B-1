function [ pred, totalTime ] = testBOW( data, labels, d, alpha, V, meanings )
%trainNN build and train the neural network

    numExamples=length(data);

    runPreds = zeros(numExamples,1);

    %Training
    totTic=tic;

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

        %Get average word meaning and predict sentence meaning
        wordMeaning=zeros(d,numWords);
        for w=1:numWords
            word=BOWlist{w,1};
            wordMeaning(:,w)=word{1};
        end
        aveMeaning=mean(wordMeaning,2);
        p = predictNode(aveMeaning,V);
        runPreds(i)=p;
    end

    totalTime = toc(totTic);
    fprintf('SGD_BOW took %f seconds (aka %f minutes) to test.\n\n',totalTime,totalTime/60);

    predd = runPreds>0.5;
    pred = zeros(size(predd));
    pred = pred + predd;

end

