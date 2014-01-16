function [logLikelihood, grad] = costFunction(weights, X, y)
%COSTFUNCTION Compute cost (log-likelihood) and gradient
%   for logistic regression

m = length(y);
n = size(weights);
mu = 0;

grad = zeros(size(weights));

x = X';
g = sigmoid(weights' * x);
l2 = mu * sum(weights.^2);

logLikelihood = sum((y .* log(g')) + ((1 - y) .* log(1 - g')));
logLikelihood = logLikelihood - l2;

for j=1:n
    grad(j) = sum((g - y') * X(:,j)) - 2*mu*weights(j);
end

end
