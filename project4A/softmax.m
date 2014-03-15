function [ pred ] = softmax( a )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

    m=max(a);
    denom = m+log(sum(exp(a-m)));
    pred = a ./ denom;

end

