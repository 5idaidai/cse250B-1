function [ err ] = squareLoss( xi, zi, num, denom )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    err = (num/denom) * (0.5*((xi - zi)^2));

end

