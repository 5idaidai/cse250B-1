function [obj,dx] = costLogLoss(t, z, alpha)
    obj = logLoss(t, z, alpha);
    dx = gradLogLoss(t, z, alpha);
end
