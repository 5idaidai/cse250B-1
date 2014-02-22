function [] = plotDocTopics(thetas)
%PLOTDOCTOPICS Plots the top 3 topics for each document in 3D space
%   Topics is 2D array/matrix with at least 3 columns (1 for each topic)
%       1 row per document


% each column is x,y,z

d=size(thetas,2);
if d >= 4
    scatter3(thetas(:,1),thetas(:,2),thetas(:,3));
else if d == 3
    scatter(thetas(:,1),thetas(:,2));
else if d == 2
    scatter(thetas(:,1),thetas(:,2));
    end
    end
end

end

