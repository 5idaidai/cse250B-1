function [ deltaRt ] = deltaRoot( t, p )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%delta value for this node
deltaRt = gradSquareLoss(t,p,1,1,1);


end

