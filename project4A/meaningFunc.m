function [ z, a ] = meaningFunc( xi, xj, W )
%meaningFunc applies the meaning function to two children

    % W (dx2d) x [xi;xj] (2dx1) = (dx1)
    a = W * [xi ; xj; 1];
    
    %h function
    z = h(a);

end

