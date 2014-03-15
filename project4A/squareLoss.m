function [ err ] = squareLoss( xi, zi, num, denom, alpha )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    err = alpha * (num/denom) * (norm(xi-zi,2)^2);

end

