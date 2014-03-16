function [ err ] = predError( t, p, alpha )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

    %err=logLoss(t,p,alpha);
    err=squareLoss(t,p,1,1,1-alpha);

end

