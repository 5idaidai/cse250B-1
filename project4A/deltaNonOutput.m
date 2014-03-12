function [ delta ] = deltaNonOutput( a, deltak, Wk, t, p )
%UNTITLED2 Summary of this function goes here
%   Equation 4 from the notes (pg. 12)

%deltak is sum of delta vectors of nodes this one feeds into, which are either just output
%or output plus non-output.

deltaP = gradLogLoss(t,p).*gradSigmoid(a);
sumk = (deltak'.* Wk)' + deltaP;
delta = hprime(a).* sumk;

end

