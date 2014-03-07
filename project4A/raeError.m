function [ err ] = raeError( k, xi, xj, ni, nj, W, b, U, c, d )
%raeError Calculate the square loss at node k
%   ni/nj = # leaves under node i/j

    nsum = ni + nj;
    nitop = ni / nsum;
    njtop = nj / nsum;
    
    mean = meaningFunc(xi, xj, k, W, b);
    z = U * mean - c;
    zi = z(1:d);
    zj = z(d+1:end);
    
    %this is with refinement #2
    err = nitop * norm(xi - zi,2) + njtop * norm(xj - zj,2);

end

