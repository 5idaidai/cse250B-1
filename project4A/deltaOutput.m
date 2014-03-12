function [ deltai ] = deltaOutput( tl, zl, tr, zr, deltak, Wk, Ul, Ur, a)
%UNTITLED Summary of this function goes here
%   Based on equation from middle of pg.11 in the notes:
%   deltai=deltaLi/deltari*hprime(ai)

deltaL = gradSquareLoss(tl,zl)*Ul;
deltaR = gradSquareLoss(tr,zr)*Ur;
deltai = (deltak*Wk + deltaL + deltaR) .* hprime(a);


end
