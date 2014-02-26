%function [] = gridsearch(file,bag,voc)
file='classic400';
load(file);
bag=classic400;
voc=classicwordlist;

fprintf('Running grid search on: %s\n',file);

numEpochs = 100;
numTopics = [2,3,4,5,10];
percs = [0.1];%[0.02,0.1,0.15,0.2];

storedBetas = 0.01;
storedAlphas = 50./numTopics;


nT = size(numTopics,1);
nP = size(percs,1);
times = zeros(nP,nT);
storedThetas = cell(nP,nT);
storedPhis = cell(nP,nT);
storedRatios = cell(nP,nT);
storedChangedThetas = cell(nP,nT);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
fprintf('\n');

kidx = 1;
pidx = 1;
for percCutOff=percs
    for k=numTopics
        alpha = storedAlphas(k);
        beta = storedBetas;
        fprintf('Running LDA with %d topics, %.04f percent cutoff\n',k,percCutOff);
        
        tic;
        [thetas,phis,ratios,alphas,betas,changeTheta ] = lda(bag, voc, words, alpha, beta, k, numEpochs, percCutOff);
        times(pidx,kidx) = toc;
        
        storedThetas(pidx,kidx) = {thetas};
        storedPhis(pidx,kidx) = {phis};
        storedRatios(pidx,kidx) = {ratios};
        storedChangedThetas(pidx,kidx) = {changeTheta};

        fprintf('LDA took %f seconds (aka %f minutes).\n\n',times(pidx,kidx),times(pidx,kidx)/60);
        
        kidx = kidx + 1;
    end
    pidx = pidx + 1;
end

temp = storedRatios(pidx-1,kidx-1);
plot(temp{1});
temp = storedPhis(pidx-1,kidx-1);
printTopKWords(temp{1},voc,10);

resultsFile = sprintf('%s_%dtopics_%depochs_grid_results.mat',file,numTopics,numEpochs);
save(resultsFile,'times','storedThetas','storedChangedThetas','storedPhis','storedRatios','storedAlphas','storedBetas','voc','numEpochs','numTopics','percs');
%end

%alphabeta_search