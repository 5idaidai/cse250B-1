function [ delta ] = deltaOutput( a, deltak, Wk )
%UNTITLED Summary of this function goes here
%   Based on equation from middle of pg.11 in the notes:
%   deltai=deltaLi/deltari*hprime(ai)

dk = (deltak'*Wk)';

%delta value for this node
delta = hprime(a) .* (dk);


end
