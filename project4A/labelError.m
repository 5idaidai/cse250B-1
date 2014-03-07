function [ err ] = labelError( label, predicted )
%labelError calculated the error the predicted label using the actual label
%   E_2 in the notes
%   Detailed explanation goes here

    %squared error
    serr = norm(label - predicted,2);
    
    %log loss error
    err = -sum(label .* log(predicted));

end

