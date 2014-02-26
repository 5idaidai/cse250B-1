%Toy data set for validation:
%6 documents, 6 unique words per doc, 2 copies of each doc 
%vocab size (18,1)
%bag size (6,18)
%topics = 3

vocab = {'cat';
    'dog';
    'bird';
    'tree';
    'grass';
    'flower';
    
    'car';
    'boat';
    'plane';
    'gas';
    'fuel';
    'diesel';
    
    'girl';
    'boy';
    'child';
    'adult';
    'mother';
    'father'};

bag = zeros(6,18);

bag(1,:)=[1;2;3;3;2;1;0;0;0;0;0;0;0;0;0;0;0;0];
bag(2,:)=[1;2;3;3;2;1;0;0;0;0;0;0;0;0;0;0;0;0];
bag(3,:)=[0;0;0;0;0;0;1;2;3;3;2;1;0;0;0;0;0;0];
bag(4,:)=[0;0;0;0;0;0;1;2;3;3;2;1;0;0;0;0;0;0];
bag(5,:)=[0;0;0;0;0;0;0;0;0;0;0;0;1;2;3;3;2;1];
bag(6,:)=[0;0;0;0;0;0;0;0;0;0;0;0;1;2;3;3;2;1];

save('toySet.mat','vocab','bag');


    
    
    