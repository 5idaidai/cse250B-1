function [ err ] = raeError( k, xi, xj, ni, nj, W, b, U, c, d )
%raeError Calculate the square loss at node k
%   ni/nj = # leaves under node i/j

    nsum = ni + nj;
    nitop = ni / nsum;
    njtop = nj / nsum;
    
    z = U * meaningFunc(xi, xj, k, W, b) - c;
    zi = z(1:d);
    zj = z(d+1:end);
    
    %this is with refinement #2, but without using the U or c vars
    %i want to use the U and c vars instead of zi, zj
    err = nitop * norm(xi - zi,2) + njtop * norm(xj - zj,2);
    
    % this is both ways of calculating error without refinement #2
    err2 = norm(([xi;xj] - U * meaningFunc(xi, xj, k, W, b) - c),2);
    err3 = norm(xi - zi,2) + norm(xj - zj,2);
    
    assert(err2==err3);

end

