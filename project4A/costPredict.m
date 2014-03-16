function [ obj,dx ] = costPredict( t, x, V, alpha )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

    [p,a] = predictNode(x, V);
    obj = squareLoss(t, p, 1, 1, 1-alpha);
    
    %dx = gradSigmoid(p);
    dx = gradPredict(t,p,a,1-alpha);
end

