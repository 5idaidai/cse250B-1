function g = gradSigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

    %g = exp(z) ./ ((exp(z) + 1).^2);
    g = sigmoid(z) .* sigmoid(-z);
end
