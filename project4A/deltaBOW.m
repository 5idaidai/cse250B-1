function [ deltaBag, deltaP ] = deltaBOW(a, t, p, V, alpha )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

%prediction deriv
deltaP = gradPredict(t,p,a,alpha);


%delta value for this node
deltaBag = (deltaP'*V)';
end

