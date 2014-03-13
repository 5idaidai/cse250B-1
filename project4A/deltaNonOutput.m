function [ delta, deltaP ] = deltaNonOutput( a, deltak, Wk, t, p )
%UNTITLED2 Summary of this function goes here
%   Equation 4 from the notes (pg. 12)


deltaP = gradLogLoss(t,p).*gradSigmoid(a);
sumk = (deltak'.* Wk)' + deltaP;
delta = hprime(a).* sumk;

end

