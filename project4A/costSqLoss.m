function [obj,dx] = costSqLoss(xi, zi, num, denom)
    obj = squareLoss(xi,zi,num,denom);
    dx = gradSquareLoss(xi,zi,num,denom);
end
