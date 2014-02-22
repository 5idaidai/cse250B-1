function [] = gridresults( filename )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

load(strcat(filename,'_grid_results.mat'));

kidx = 1;
pidx = 1;
for percCutOff=percs
    for k=numTopics
        plotDocTopics(storedThetas(pidx,kidx));
        printTopKWords(storedPhis(pidx,kidx),voc,10);
        pause;
    end
end

end

