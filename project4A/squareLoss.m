function [ err ] = squareLoss( xi, zi, num, denom, alpha )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    err = alpha * (num/denom) * 0.5*(norm(xi-zi)^2);

end

