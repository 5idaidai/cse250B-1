function [ err ] = squareLoss( xi, xj, ni, nj, zi, zj )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    nsum = ni + nj;
    nitop = ni / nsum;
    njtop = nj / nsum;
    err = nitop * (0.5*((xi - zi)^2)) + njtop * (0.5*((xj - zj)^2));

end

