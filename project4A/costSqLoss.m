function [obj,dx] = costSqLoss(xi, zi, num, denom,alpha)
    obj = squareLoss(xi,zi,num,denom,alpha);
    dx = gradSquareLoss(xi,zi,num,denom,alpha);
end
