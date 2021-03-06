function [ delta, deltaP ] = deltaInput( a, deltak, Wk, V, t, p, alpha )
%UNTITLED2 Summary of this function goes here
%   Equation 4 from the notes (pg. 12)

%from parent node
deltapar = (deltak'*Wk)';

%prediction deriv
deltaP = gradPredict(t,p,V*a,alpha);
deltaPr = (deltaP'*V)';

delta = deltapar + deltaPr;

end

