function [ z, a ] = meaningFunc( xi, xj, W, atRoot )
%meaningFunc applies the meaning function to two children

    % W (dx2d) x [xi;xj] (2dx1) = (dx1)
    a = W * [xi ; xj];
    
    if atRoot
        z=a;
    else
        %h function
        z = h(a);
    end

end

