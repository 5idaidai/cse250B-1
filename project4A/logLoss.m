function [ err ] = logLoss( label, predicted )
%labelError calculated the error the predicted label using the actual label
%   E_2 in the notes
%   Detailed explanation goes here

    %log loss error
    %err = -sum(label .* log(predicted));
    err = -mean(label.*log(predicted) + (1-label).*(1-log(predicted)));

end

