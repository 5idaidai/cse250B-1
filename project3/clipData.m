load ('20Newsgroups.mat');

bag=fea;
voc=vocab;

x=zeros([800,26214]);
vocabShort=cell(10000,2);
vtemp=voc;

for d=1:800
    x(d,:)=bag(d,:);
end

feaShort=sparse(x);

for w=1:26214
    vtemp{w,2}=sum(x(:,w));
end

vttemp=sortrows(vtemp,-2); 
vocabShort(:)=vttemp((1:10000),:);

save ('20NewsgroupsShort.mat','feaShort','vocabShort');
    

