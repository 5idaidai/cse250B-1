% prompt = 'Enter 1 to use the Classic400 dataset \nEnter 2 to use the 20Newsgroups dataset \n';
% dataset = input(prompt);
% 
% if dataset==1
%     file='classic400';
%     load(file);
%     bag=classic400;
%     voc=classicwordlist;
% else if dataset==2
%         file='20NewsgroupsShort';
%         load(file);
%         bag=feaShort;
%         voc=vocabShort;
%     else if dataset>2
%             print 'Not a valid dataset';
%             break
%         end
%     end
% end
    file='classic400';
    %file='toySet';
    load(file);
    bag=classic400;
    voc=classicwordlist;
    %bag=bag;
    %voc=vocab;

fprintf('Running epoch num search on: %s\n',file);

nEpoch = 1;
numEpochList = [150];%[100,200,300,500];
numTopics = 10;
percCutOff = 0.1;

storedBetas = 0.01;
storedAlphas = 50./numTopics;

nT = size(numEpochList,1);
times = zeros(nT);
storedThetas = cell(nT);
storedPhis = cell(nT);
storedRatios = cell(nT);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
fprintf('\n');

pidx = 1;
for numEpochs=numEpochList
    fprintf('Running LDA with %d topics, %.04f percent cutoff\n',numTopics,percCutOff);
    burnIn = ceil(max(numEpochs/10,1));

    tic;
    [thetas,phis,ratios,alphas,betas] = lda(bag, voc, words, storedAlphas, storedBetas, numTopics, numEpochs, percCutOff, burnIn, nEpoch);
    times(pidx) = toc;

    storedThetas(pidx) = {thetas};
    storedPhis(pidx) = {phis};
    storedRatios(pidx) = {ratios};

    fprintf('LDA took %f seconds (aka %f minutes).\n\n',times(pidx),times(pidx)/60);
    pidx = pidx + 1;
end

figure;
plotDocTopics(thetas);

figure;
plot(ratios);

printTopKWords(phis,voc,10);

resultsFile = sprintf('%s_%dtopics_200epoch_results.mat',file,numTopics);
%save(resultsFile,'times','storedThetas','storedPhis','storedRatios','storedAlphas','storedBetas','voc','numTopics','numEpochList');
