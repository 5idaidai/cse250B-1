function [ delta, deltaP, dLdGammaL, dLdGammaR ] = deltaOutput( t, p, tl, zl, tr, zr, deltak, Wk, Ul, Ur, a, V, topL, topR, bottom)
%UNTITLED Summary of this function goes here
%   Based on equation from middle of pg.11 in the notes:
%   deltai=deltaLi/deltari*hprime(ai)

%RAE derivs
dLdGammaL = gradSquareLoss(tl,zl,topL,bottom);
dLdGammaR = gradSquareLoss(tr,zr,topR,bottom);
deltaL = Ul*dLdGammaL;
deltaR = Ur*dLdGammaR;

%Prediction derivs
deltaP = gradLogLoss(t,p).*gradSigmoid(p);
deltaPr = (deltaP'*V)';

dk = (deltak'*Wk)';

%delta value for this node
delta = hprime(a) .* (dk + deltaL + deltaR + deltaPr);


end
