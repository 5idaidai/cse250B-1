%function [] = gridsearch(file,bag,voc)
file='classic400';
load(file);
bag=classic400;
voc=classicwordlist;

fprintf('Running grid search on: %s\n',file);

numEpochs = 300;
numTopics = 5;
percCutOff = 0.1;

posbetas = [0.1, 0.05, 0.01];
posalphas = [50/numTopics, 25/numTopics, 10/numTopics, 5/numTopics];


nT = size(posbetas,1);
nP = size(posalphas,1);
times = zeros(nP,nT);
storedThetas = cell(nP,nT);
storedPhis = cell(nP,nT);
storedRatios = cell(nP,nT);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
fprintf('\n');

kidx = 1;
pidx = 1;
for alpha=posalphas
    for beta=posbetas
        fprintf('Running LDA with %f alpha, %f beta\n',alpha,beta);
        
        tic;
        [thetas,phis,ratios,alphas,betas] = lda(bag, voc, words, alpha, beta, numTopics, numEpochs, percCutOff);
        times(pidx,kidx) = toc;
        
        storedThetas(pidx,kidx) = {thetas};
        storedPhis(pidx,kidx) = {phis};
        storedRatios(pidx,kidx) = {ratios};

        fprintf('LDA took %f seconds (aka %f minutes).\n\n',times(pidx,kidx),times(pidx,kidx)/60);
        
        kidx = kidx + 1;
    end
    pidx = pidx + 1;
end

resultsFile = strcat(file,'_alphabetasearch_grid_results.mat');
save(resultsFile,'times','storedThetas','storedPhis','storedRatios','posalphas','posbetas','voc','numEpochs','numTopics','percCutOff');
%end