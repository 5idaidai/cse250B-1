function [ delta, deltaP ] = deltaInput( deltak, Wk, V, t, p, alpha )
%UNTITLED2 Summary of this function goes here
%   Equation 4 from the notes (pg. 12)

%from parent node
deltapar = (deltak'*Wk)';

%prediction deriv
left=gradLogLoss(t,p);
right=gradSigmoid(p);
deltaP = left.*right;
deltaP = (1-alpha)*deltaP;
deltaPr = (deltaP'*V)';

delta = deltapar + deltaPr;

end

