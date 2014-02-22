
load('classic400.mat');
%load('20Newsgroups.mat');

bag=classic400;
voc=classicwordlist;

numTopics = 4;
numEpochs = 30;
percCutOff = 0.15;
    
tic;
[thetas,phis] = lda(numTopics, bag, voc, numEpochs, percCutOff);
TimeSpent = toc;

fprintf('LDA took %f seconds (aka %f minutes).\n\n',TimeSpent,TimeSpent/60);

plotDocTopics(thetas);
printTopKWords(phis,voc,10);