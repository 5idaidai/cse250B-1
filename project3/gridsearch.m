function [] = gridsearch(file,bag,voc)

fprintf('Running grid search on: %s\n',file);

numEpochs = 500;
numTopics = [2,3,4,5,10,15,20];
percs = [0.1];%[0.02,0.1,0.15,0.2];


nT = size(numTopics,1);
nP = size(percs,1);
times = zeros(nP,nT);
storedThetas = cell(nP,nT);
storedPhis = cell(nP,nT);
storedRatios = cell(nP,nT);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
fprintf('\n');

kidx = 1;
pidx = 1;
for percCutOff=percs
    for k=numTopics
        fprintf('Running LDA with %d topics, %.04f percent cutoff\n',k,percCutOff);
        
        tic;
        [thetas,phis,ratios,alphas,betas] = lda(bag, voc, words, k, numEpochs, percCutOff);
        times(pidx,kidx) = toc;
        
        storedThetas(pidx,kidx) = {thetas};
        storedPhis(pidx,kidx) = {phis};
        storedRatios(pidx,kidx) = {ratios};

        fprintf('LDA took %f seconds (aka %f minutes).\n\n',times(pidx,kidx),times(pidx,kidx)/60);
        
        kidx = kidx + 1;
    end
    pidx = pidx + 1;
end

resultsFile = strcat(file,'_xepochs_grid_results.mat');
save(resultsFile,'times','storedThetas','storedPhis','storedRatios','voc','numEpochs','numTopics','percs');
end