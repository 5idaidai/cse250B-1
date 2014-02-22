
load('classic400.mat');
%load('20Newsgroups.mat');
numTopics = 4;
numEpochs = 30;

[thetas,phis] = lda(numTopics, classic400, classicwordlist, numEpochs);
