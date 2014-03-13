function [ deltaRt, deltaP, dLdGammaL, dLdGammaR ] = deltaRoot( tl, zl, tr, zr, Ul, Ur, a, t, p, V, topL, topR, bottom)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

dLdGammaL = gradSquareLoss(tl,zl,topL,bottom);
dLdGammaR = gradSquareLoss(tr,zr,topR,bottom);
deltaL = dLdGammaL'*Ul;
deltaR = dLdGammaR'*Ur;
deltaP = gradLogLoss(t,p).*gradSigmoid(a);
deltaRt = (deltaP'*V)' + deltaL' + deltaR';


end

