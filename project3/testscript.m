
%load('classic400.mat');
load('20Newsgroups.mat');

%bag=classic400;
%voc=classicwordlist;

bag=fea;
voc=vocab;

numTopics = 10;
numEpochs = 1;
percCutOff = 0.15;


words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
    
tic;
[thetas,phis] = lda(bag, voc, words, numTopics, numEpochs, percCutOff);
TimeSpent = toc;

fprintf('LDA took %f seconds (aka %f minutes).\n\n',TimeSpent,TimeSpent/60);

%plotDocTopics(thetas);
printTopKWords(phis,voc,10);