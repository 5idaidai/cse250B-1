
load('classic400.mat');
%load('20Newsgroups.mat');
numTopics = 4;
numEpochs = 30;

lda(numTopics, classic400, classicwordlist, numEpochs);
