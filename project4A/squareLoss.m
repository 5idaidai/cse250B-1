function [ err ] = squareLoss( xi, xj, ni, nj, zi, zj )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    nsum = ni + nj;
    nitop = ni / nsum;
    njtop = nj / nsum;
    err = nitop * norm(xi - zi,2) + njtop * norm(xj - zj,2);

end

