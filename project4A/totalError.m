function [ totErr ] = totalError( sentTree, outputItr, innerItr, alpha )
%totalError total error for one sentence s with label t
%   Detailed explanation goes here

    outputErr = zeros(num(outputItr));
    for idx=[1,outputItr]
        node=sentTree.get(idx);
        E1=node{8};
        E2=node{13};
        outputErr(idx) = alpha*E1 + (1-alpha)*E2;
    end
    
    innerErr = zeros(num(innerItr));
    for idx=innerItr
    node=sentTree.get(idx);
    E2=node{13};
    innerErr(idx) = (1-alpha)*E2;
    end
    
    totErr = sum(outerErr,innerErr);

end

