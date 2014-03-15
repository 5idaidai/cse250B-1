function [ grad ] = gradLogLoss(t,r,alpha)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    grad = (1-alpha)*(r-t);

end
