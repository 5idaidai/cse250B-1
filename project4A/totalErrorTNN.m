function [ totErr ] = totalErrorTNN( outputItr, numDiffTree, W )
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here

    outputErr = zeros(length(outputItr),1);
    for idx=[outputItr];
        node = numDiffTree.get(idx);
        E2 = node{13};
        outputErr(idx) = E2;
    end
    
    totErr = sum(outputErr);


end

