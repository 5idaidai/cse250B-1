function [ deltak ] = deltak( t, z, x, top, bottom, a )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%kindex = number of nodes above this one

if kindex==1
    deltak = deltaOutput(t, z, x, top, bottom, a);
else
    deltak = deltaOutput(t, z, x, top, bottom, a) + kindex(z-t);
end

    

end

