function [ predicted ] = predictNode( xk, V )
%predict Calculated predicted label based on meaning vector of node

    %sigmoid because it's binary?
    predicted = sigmoid(V * xk);

end

