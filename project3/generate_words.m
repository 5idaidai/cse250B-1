load('classic400.mat');

format shortg;
clock

numWords = sum(sum(classic400'));
words = get_words(classic400', classicwordlist, numWords, size(classic400,1));

clock

save('classic400WordOcc.mat',words,numWords);




load('20Newsgroups.mat');

format shortg;
clock

numWords = sum(sum(fea'));
words = get_words(fea', vocab, numWords, size(fea,1));

clock

save('20NewsWordOcc.mat',words,numWords);