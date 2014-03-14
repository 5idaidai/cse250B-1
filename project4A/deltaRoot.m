function [ deltaRt, deltaP, dLdGammaL, dLdGammaR ] = deltaRoot( tl, zl, tr, zr, Ul, Ur, a, t, p, V, topL, topR, bottom)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%RAE derivs
dLdGammaL = gradSquareLoss(tl,zl,topL,bottom);
dLdGammaR = gradSquareLoss(tr,zr,topR,bottom);
deltaL = Ul*dLdGammaL;
deltaR = Ur*dLdGammaR;

%prediction deriv
left=gradLogLoss(t,p);
right=gradSigmoid(p);
deltaP = left.*right;
deltaPr = (deltaP'*V)';

%delta value for this node
deltaRt = hprime(a) .* (deltaPr + deltaL + deltaR);


end

