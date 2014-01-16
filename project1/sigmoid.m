function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

if (size(z,1) > 1 || size(z,2) > 1)
	for i=1:size(z,1)
		if (size(z,2) > 1)
			for j=1:size(z,2)
				g(i,j) = 1 / (1 + (exp(- z(i,j))));
            end
		else 
			g(i) = 1 / (1 + (exp(- z(i))));
        end
    end
else
	g = 1 / (1 + (e ^ - z));
end

end
