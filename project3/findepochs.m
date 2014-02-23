file='classic400';
load(file);

bag=classic400;
voc=classicwordlist;

fprintf('Running epoch num search on: %s\n',file);

numEpochList = [100,200,300,500,1000];
numTopics = 20;
percCutOff = 0.1;

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

    tic;
    [thetas,phis,ratios] = lda(bag, voc, words, numTopics, numEpochs, percCutOff);
    times(pidx) = toc;

    storedThetas(pidx) = {thetas};
    storedPhis(pidx) = {phis};
    storedRatios(pidx) = {ratios};

    fprintf('LDA took %f seconds (aka %f minutes).\n\n',times(pidx),times(pidx)/60);
    pidx = pidx + 1;
end

resultsFile = strcat(file,'_epoch_results.mat');
save(resultsFile,'times','storedThetas','storedPhis','storedRatios','voc','numTopics','percs');
