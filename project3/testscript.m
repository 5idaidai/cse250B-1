
load('classic400.mat');
%load('20Newsgroups.mat');

bag=classic400;
voc=classicwordlist;

numTopics = 4;
numEpochs = 30;
    
[thetas,phis] = lda(numTopics, bag, voc, numEpochs);

plotDocTopics(thetas);
printTopicWords(phis,voc,numTopics);