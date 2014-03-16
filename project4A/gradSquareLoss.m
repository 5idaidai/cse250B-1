function [ grad ] = gradSquareLoss( x, z, top, bottom, alpha )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

grad = alpha*2*(top/bottom)*(z-x);


end

