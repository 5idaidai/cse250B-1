
load('classic400.mat');
%load('20Newsgroups.mat');
numTopics = 4;

lda(numTopics, classic400, classicwordlist);
