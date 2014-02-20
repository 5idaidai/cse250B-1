function [ output_args ] = innergibbs(alpha, beta, q, n)
%GIBBS Equation 5 in the notes
%   drawing a random number uniformly between 0 and
% 1, and using it to index into the unit interval which is divided into subintervals
% of length p

    left = q + beta;
    right = n + alpha;
    
    denomLeft = sum(left);
    denomRight = sum(right);
    
    P = (left/denomLeft)*(right/denomRight);
    
    r = rand;
    % r is index into 0-1 interval, subdivided into P(i) intervals

end

