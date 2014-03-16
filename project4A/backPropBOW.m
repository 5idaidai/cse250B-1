function [ dV, BOWlist, dMeaning ] = backPropBOW( BOWlist, V, alpha, t, meanings, numWords)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    

    %each node contains the following:
    % 1: meaning vector
    % 2: z, predicted label
    % 3: a, activation
    % 4: delta vector
    % 5: word index
    % 6: E2, prediction error
    
    dV = zeros(size(V));
    dMeaning = zeros(size(meanings));
    
    % Calculate deltas for output nodes
    for i=1:length(numWords)
        node = BOWlist{i,1};
        a = node{3};
        p = node{2};
        [ deltaBag, deltaP ] = deltaBOW( a, t, p, V, alpha );
        
        BOWlist{i,1}{4,1} = deltaBag;
        deltaV = deltaP*node{1}';
        dV = dV + deltaV;
    end 
    
    % Calculate deltas for input nodes to update meaning
    for i=1:length(numWords)
        node = BOWlist{i,2};
        parent = BOWlist{i,1};
        delta = parent{4};
        deltaV = deltaP*node{1}';
        dV = dV + deltaV;
        
        dMeaning(:,node{5}) = dMeaning(:,node{5}) + delta;
    end      

end

