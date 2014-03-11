function [obj,dx] = costLogLoss(t, z)
    obj = logLoss(t, z);
    dx = gradLogLoss(t, z);
end
