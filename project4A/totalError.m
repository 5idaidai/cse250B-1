function [ totErr ] = totalError( outputItr, innerItr, alpha, numDiffTree, lambda, W, U, V )
%totalError total error for one sentence s with label t
%   Detailed explanation goes here

    outputErr = zeros(length(outputItr),1);
    for idx=[1,outputItr];
        node = numDiffTree.get(idx);
        E1 = node{8};
        E2 = node{13};
        outputErr(idx) = E2;%alpha*E1 + (1-alpha)*E2;
    end
    
    innerErr = zeros(length(innerItr),1);
    for idx=innerItr
        node = numDiffTree.get(idx);
        E2 = node{13};
        innerErr(idx) = (1-alpha)*E2;
    end
    
    totErr = sum(outputErr)+sum(innerErr) + (lambda(1:3)./2)*([norm(W),norm(U),norm(V)].^2)';

end

