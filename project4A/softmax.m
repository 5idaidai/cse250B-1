function [ pred ] = softmax( a )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    denom = sum(exp(a));
    pred = a ./ denom;

end

