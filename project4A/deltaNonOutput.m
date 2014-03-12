function [ delta ] = deltaNonOutput( a, deltak, Vk, T )
%UNTITLED2 Summary of this function goes here
%   Equation 4 from the notes (pg. 12)

%deltak is sum of delta vectors of nodes this one feeds into, which are either just output
%or output plus #non-output.
%
%deltak = deltaOutput + sum(deltaNonOutput parents)

%T = ??

sumk = deltak^T * Vk;
delta = hprime(a).*(sumk)^T;

end

