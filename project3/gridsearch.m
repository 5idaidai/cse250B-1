
load('classic400.mat');
%load('20Newsgroups.mat');

bag=classic400;
voc=classicwordlist;

numEpochs = 1;
numTopics = [2,3,4];%,5,10,15,20];
percs = [0.02,0.1];%,0.15,0.2];


nT = size(numTopics,1);
nP = size(percs,1);
times = zeros(nP,nT);
storedThetas = cell(nP,nT);
storedPhis = cell(nP,nT);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
fprintf('\n');

kidx = 1;
pidx = 1;
for percCutOff=percs
    for k=numTopics
        fprintf('Running LDA with %d topics, %f percent cutoff\n',k,percCutOff);
        
        tic;
        [thetas,phis] = lda(bag, voc, words, k, numEpochs, percCutOff);
        times(pidx,kidx) = toc;
        
        storedThetas(pidx,kidx) = {thetas};
        storedPhis(pidx,kidx) = {phis};

        fprintf('LDA took %f seconds (aka %f minutes).\n\n',times(pidx,kidx),times(pidx,kidx)/60);

        %plotDocTopics(thetas);
        %printTopKWords(phis,voc,10);
        
        kidx = kidx + 1;
    end
    pidx = pidx + 1;
end