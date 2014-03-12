function [ deltaRoot, deltaP ] = deltaRoot( tl, zl, tr, zr, Ul, Ur, a, t, p)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

deltaL = gradSquareLoss(tl,zl)*Ul;
deltaR = gradSquareLoss(tr,zr)*Ur;
deltaP = gradLogLoss(t,p).*gradSigmoid(a);
deltaRoot = deltaP + deltaL + deltaR;


end

