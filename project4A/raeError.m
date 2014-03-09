function [ err, zi, zj ] = raeError( xk, xi, xj, ni, nj, U, c, d )
%raeError Calculate the square loss at node k
%   ni/nj = # leaves under node i/j
%   E_1 in the notes
    
    %mean = meaningFunc(xi, xj, k, W, b);
    z = U * xk - c;
    zi = z(1:d);
    zj = z(d+1:end);
    
    %this is with refinement #2
    err = squareLoss(xi,xj,ni,nj,zi,zj);

end

