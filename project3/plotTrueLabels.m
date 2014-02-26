function [] = plotTrueLabels(truelabels)

labelsPerDoc = zeros(400,3);
for i=1:400
    idx=truelabels(:,i);
    labelsPerDoc(i,idx)=1;
end

C = 'm';
S=300;
scatter3(labelsPerDoc(:,1),labelsPerDoc(:,2),labelsPerDoc(:,3),S,C,'filled'),view(-45,10);
end

