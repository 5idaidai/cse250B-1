function [ delta, deltaP, dLdGammaL, dLdGammaR ] = deltaOutput( t, p, tl, zl, tr, zr, deltak, Wk, Ul, Ur, a, V, topL, topR, bottom)
%UNTITLED Summary of this function goes here
%   Based on equation from middle of pg.11 in the notes:
%   deltai=deltaLi/deltari*hprime(ai)

dLdGammaL = gradSquareLoss(tl,zl,topL,bottom);
dLdGammaR = gradSquareLoss(tr,zr,topR,bottom);
deltaL = dLdGammaL'*Ul;
deltaR = dLdGammaR'*Ur;
deltaP = gradLogLoss(t,p).*gradSigmoid(a);
delta = ((deltak'*Wk)' + deltaL' + deltaR' + (deltaP'*V)') .* hprime(a);


end
