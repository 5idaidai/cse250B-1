function [] = plotDocTopics(thetas)
%PLOTDOCTOPICS Plots the top 3 topics for each document in 3D space
%   Topics is 2D array/matrix with at least 3 columns (1 for each topic)
%       1 row per document


% each column is x,y,z

M=size(thetas,1);
d=size(thetas,2);
C = repmat([1 2 4],size(thetas,2));
S=20;

sorted = zeros(size(thetas));
[sorted,sidx]=sort(thetas,2,'descend');

if d >= 4
    scatter3(sorted(:,1),sorted(:,2),sorted(:,3));
else if d == 3
    scatter3(sorted(:,1),sorted(:,2));
else if d == 2
    scatter3(sorted(:,1),sorted(:,2));
    end
    end
end

end
