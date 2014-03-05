
load('codeDataMoviesEMNLP/data/rt-polaritydata/vocab.mat');

%init meaning vectors for each word to random values
d = 20;
meanings = rand(d,size(words,2));

%init W and b randomly
W = rand(d,2*d);
b = rand(d,1);