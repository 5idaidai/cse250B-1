function [ err ] = totalError( sent, nonLeafTree, numNonLeafNodes, label, W, b, U, c, V, alpha )
%totalError total error for one sentence s with label t
%   Detailed explanation goes here

    nodeErr = ones(numNonLeafNodes);
    for k=1:numNonLeafNodes
        predicted = predictNode(xk,V);
        nodeErr(k) = alpha * raeError(k,xi,xj,ni,nj,W,b,U,c,d) + (1-alpha)*labelError(label,predicted);
    end
    
    err = sum(nodeErr);

end

