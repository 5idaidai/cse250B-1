function [ err ] = squareLoss( xi, zi, num, denom )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    err = (num/denom) * (norm(xi-zi,2)^2);

end

