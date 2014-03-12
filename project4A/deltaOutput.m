function [ deltai ] = deltaOutput( t, z, x, top, bottom, a)
%UNTITLED Summary of this function goes here
%   Based on equation from middle of pg.11 in the notes:
%   deltai=deltaLi/deltari*hprime(ai)

deltaL = gradLogLoss(t,z)*2 + gradSquareLoss(x,z,top,bottom);
%r = vector value of output node i
%deltar = ??
deltai = deltaL/deltar * hprime(a);


end
