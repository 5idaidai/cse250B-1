function [ grad ] = gradSquareLoss( x, z, top, bottom, alpha )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

grad = alpha*(top/bottom)*(x-z);


end

