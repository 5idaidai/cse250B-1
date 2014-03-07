function [ xk ] = meaningFunc( xi, xj, W, b )
%meaningFunc applies the meaning function to two children

    % W (dx2d) x [xi;xj] (2dx1) = (dx1)
    cross = W * [xi ; xj];
    temp = cross + b;
    
    %h function
    xk = h(temp);

end

