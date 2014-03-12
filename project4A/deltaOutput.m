function [ deltai, deltaP, dLdGammaL, dLdGammaR ] = deltaOutput( t, p, tl, zl, tr, zr, deltak, Wk, Ul, Ur, a, V)
%UNTITLED Summary of this function goes here
%   Based on equation from middle of pg.11 in the notes:
%   deltai=deltaLi/deltari*hprime(ai)

dLdGammaL = gradSquareLoss(tl,zl);
dLdGammaR = gradSquareLoss(tr,zr);
deltaL = dLdGammaL'*Ul;
deltaR = dLdGammaR'*Ur;
deltaP = gradLogLoss(t,p).*gradSigmoid(a);
deltai = ((deltak'*Wk)' + deltaL' + deltaR' + (deltaP'*V)') .* hprime(a);


end
