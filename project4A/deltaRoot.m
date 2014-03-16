function [ deltaRt, deltaP, dLdGammaL, dLdGammaR ] = deltaRoot( tl, zl, tr, zr, Ul, Ur, a, t, p, V, topL, topR, bottom, alpha)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%RAE derivs
dLdGammaL = gradSquareLoss(tl,zl,topL,bottom,alpha);
dLdGammaR = gradSquareLoss(tr,zr,topR,bottom,alpha);
deltaL = Ul*dLdGammaL;
deltaR = Ur*dLdGammaR;

%prediction deriv
deltaP = gradPredict(t,p,V*a,alpha);
deltaPr = (deltaP'*V)';

%delta value for this node
deltaRt = hprime(a) .* (deltaPr + deltaL + deltaR);


end

