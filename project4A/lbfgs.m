function [] = lbfgs()
%LBFGS Run L-BFGS on the dataset
%   Run Stochastic gradient descent with L2 regularization
%       on the dataset. Use grid search to find values for
%       the hyperparameters
    fprintf('Running L-BFGS on the dataset\n');

    %using minFunc from: Mark Schmidt
    % http://www.di.ens.fr/~mschmidt/Software/minFunc.html
    options = [];
    options.Method = 'lbfgs';
    varargin = [];
    
%    [theta, cost] = ...
%	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
%[theta2, cost2] = ...
%	minFunc(@(t)(costFunction(t, X, y)), initial_theta, options2);

    
    [x,f,exitflag,output] = minFunc(@(t)(costFunction(t, X, y, mu)),initial_theta,options,varargin)
end

