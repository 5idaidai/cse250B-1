function [ err ] = raeError( k, xi, xj, ni, nj, zi, zj, W, b, U, c )
%raeError Calculate the square loss at node k
%   Detailed explanation goes here

    nsum = ni + nj;
    nitop = ni / nsum;
    njtop = nj / nsum;
    
    %this is with refinement #2, but without using the U or c vars
    %i want to use the U and c vars instead of zi, zj
    err = nitop * abs(xi - zi)^2 + njtop * abs(xj - zj)^2;
    
    % this is both ways of calculating error without refinement #2
    err2 = abs([xi;xj] - U * meaningFunc(xi, xj, k, W, b) - c)^2;
    err3 = abs(xi - zi)^2 + abs(xj - zj)^2;
    
    assert(err2==err3);

end

