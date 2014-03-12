function [ deltaRoot, deltaP, dLdGammaL, dLdGammaR ] = deltaRoot( tl, zl, tr, zr, Ul, Ur, a, t, p, V )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

dLdGammaL = gradSquareLoss(tl,zl);
dLdGammaR = gradSquareLoss(tr,zr);
deltaL = dLdGammaL'*Ul;
deltaR = dLdGammaR'*Ur;
deltaP = gradLogLoss(t,p).*gradSigmoid(a);
deltaRoot = (deltaP'*V)' + deltaL' + deltaR';


end

