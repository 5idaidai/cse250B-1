function [ err, zi, zj ] = raeError( xk, xi, xj, ni, nj, U, c, d )
%raeError Calculate the square loss at node k
%   ni/nj = # leaves under node i/j
%   E_1 in the notes
    
    %mean = meaningFunc(xi, xj, k, W, b);
    z = U * xk - c;
    zi = z(1:d);
    zj = z(d+1:end);
    
    %this is with refinement #2
    nsum = ni + nj;
    err = squareLoss(xi,zi,ni,nsum) + squareLoss(xj,zj,nj,nsum);

end

