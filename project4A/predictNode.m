function [ predicted ] = predictNode( xk, V )
%predict Calculated predicted label based on meaning vector of node

    %predicted = sigmoid(V * xk);
    predicted = softmax(V * xk);

end

