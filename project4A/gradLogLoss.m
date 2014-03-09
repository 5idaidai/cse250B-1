function [ grad ] = gradLogLoss( t, z)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

grad = -t/z + (1-t)/(1-z);


end
