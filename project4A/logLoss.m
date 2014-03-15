function [ err ] = logLoss( y, y_pred, alpha )
%labelError calculated the error the predicted label using the actual label
%   E_2 in the notes
%   Detailed explanation goes here

    %log loss error
    %err = (1-alpha)*-sum(y.*log(y_pred)+(1-y).*log(1-y_pred));
    err = (1-alpha)*-sum(y.*log(y_pred));

end

