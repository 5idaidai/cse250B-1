function [ predicted, a ] = predictNode( xk, V )
%predict Calculated predicted label based on meaning vector of node

    a=V*xk;
    predicted = sigmoid(a);
    %predicted = softmaxNN(a);

end

