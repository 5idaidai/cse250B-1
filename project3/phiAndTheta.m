function [theta, phi] = phiAndTheta(q,n,m,k,vocabSize)
%Recovers phi and theta

theta = n;
for i=1:m
    r1sum=sum(n(i,:));
    for j=1:k
        theta(i,j) = n(i,j)/r1sum;
    end
end

phi = q;
for a=1:k
    r2sum=sum(q(a,:));
    for b=1:vocabSize
        phi(a,b) = q(a,b)/r2sum;
    end
end
end



