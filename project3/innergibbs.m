function [ output_args ] = innergibbs(alpha, beta, q, n)
%GIBBS Summary of this function goes here
%   Detailed explanation goes here

    left = q + beta;
    right = n + alpha;
    
    denomLeft = sum(left);
    denomRight = sum(right);
    
    P = (left/denomLeft)*(right/denomRight);
    
    r = rand;
    % r is index into 0-1 interval, subdivided into P(i) intervals

end

