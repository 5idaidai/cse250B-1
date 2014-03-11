function [obj,dx] = costSigmoid(a)
    obj = sigmoid(a);
    dx = gradSigmoid(a);
end
