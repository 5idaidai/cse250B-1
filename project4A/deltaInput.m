function [ delta, deltaP ] = deltaInput( a, deltak, Wk, trainInput )
%UNTITLED2 Summary of this function goes here
%   Equation 4 from the notes (pg. 12)

delta = (deltak'*Wk)';

if trainInput
end

end

