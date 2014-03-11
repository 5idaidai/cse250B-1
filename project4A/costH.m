function [obj,dx] = costH(a)
    obj = h(a);
    dx = hprime(a);
end
