function [ grad ] = gradPredict( t, r, a, alpha )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

    %gradT = gradLogLoss(t, r, alpha);
    gradT = gradSquareLoss(t, r, 1, 1, 1-alpha);
    %gradSig = gradSoftmax(a);
    gradSig = gradSigmoid(a);
    
    grad = gradT.*gradSig;
    %grad = gradT;

end