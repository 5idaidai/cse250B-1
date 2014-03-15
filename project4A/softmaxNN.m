function [ pred ] = softmaxNN( a )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    denom = sum(exp(a));
    pred = exp(a) ./ denom;

end

