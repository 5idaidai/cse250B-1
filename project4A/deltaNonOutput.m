function [ delta, deltaP ] = deltaNonOutput( a, deltak, Wk, V, t, p, alpha )
%UNTITLED2 Summary of this function goes here
%   Equation 4 from the notes (pg. 12)

deltaP = gradLogLoss(t,p).*gradSigmoid(p);
deltaP = (1-alpha)*deltaP;
deltaPr = (deltaP'*V)';

dk = (deltak'*Wk)';

sumk = dk + deltaPr;
delta = hprime(a) .* sumk;

end

