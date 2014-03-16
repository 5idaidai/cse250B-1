function [ g ] = gradSoftmax( z )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

    g = softmaxNN(z) .* softmaxNN(-z);

end

