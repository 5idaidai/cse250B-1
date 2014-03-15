function [ grad ] = gradPredict( t, r, alpha )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

    gradT = gradLogLoss(t, r, alpha);
    
    grad = gradT.*gradSoftmax(r);
    %grad = gradT;

end