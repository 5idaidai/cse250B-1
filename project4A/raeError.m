function [ err, zl, zr, el, er ] = raeError( xk, xi, xj, ni, nj, U, d )
%raeError Calculate the square loss at node k
%   ni/nj = # leaves under node i/j
%   E_1 in the notes
    
    %mean = meaningFunc(xi, xj, k, W, b);
    z = U * [xk; -1];
    zl = z(1:d);
    zr = z(d+1:end);
    
    %this is with refinement #2
    nsum = ni + nj;
    el = squareLoss(xi,zl,ni,nsum);
    er = squareLoss(xj,zr,nj,nsum);
    err = el + er;

end

