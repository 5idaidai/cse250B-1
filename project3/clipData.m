load ('20Newsgroups.mat');

bag=fea;
voc=vocab;

x=zeros([800,26214]);
vocabShort=cell(10000,2);

r=randi(18846,800);

for d=1:800
    x(d,:)=bag(r(d),:);
end

for w=1:10000
    vocabShort(w,:)=voc(w,:);
end

feaShort=sparse(x);
save ('20NewsgroupsShort.mat','feaShort','vocabShort');
    

