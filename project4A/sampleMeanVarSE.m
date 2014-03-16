function [ mu, sigma, se ] = sampleMeanVarSE( sample )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

    n = length(sample);
    mu = mean(sample);
    sigma = var(sample);%(1/n) * sum(sample - mu);
    se = sqrt(sigma/n);

end

