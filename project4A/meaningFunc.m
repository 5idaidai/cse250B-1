function [ z, a ] = meaningFunc( xi, xj, W, b )
%meaningFunc applies the meaning function to two children

    % W (dx2d) x [xi;xj] (2dx1) = (dx1)
    cross = W * [xi ; xj; 1];
    a = cross;% + b;
    
    %h function
    z = h(a);

end

