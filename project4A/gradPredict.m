function [ grad ] = gradPredict( t, r, alpha )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

    gradT = (1-alpha)*(t - r);
    
    grad = -gradT.*gradSigmoid(r);

end