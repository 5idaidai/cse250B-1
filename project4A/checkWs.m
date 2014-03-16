%Script to test the backProp and numDiff W matrices based on Quiz 9 model.



sent = ones(1,3);
numWords=length(sent);

%hyperparameters
trainInput = 0;%don't train input for now
d = 2;
lambda = [1e-05, 0.0001, 1e-07, 0.01];
alpha = 0.4;

%init meaning vectors for each word to one
meanings = ones(1,3);

%init W randomly
W = [.2,.4];

%init U and c for backpropagation
U = ones(2,1);

%init V for prediction
% v1 = rand(1,d);
% v2 = 1-v1;
% V = zeros(2,d);
% V(1,:)=v1;
% V(2,:)=v2;
V = 1;

totTic=tic;



%skip sentences of less than 2 words because our our neural nets
%are defined for these
if numWords>=2
    t=0;

    %build up sentence binary tree, and perform feed forward
    %algorithm at the same time
    [sentTree, outputItr, innerItr, inputItr] = buildTree(sent, meanings, numWords, W, U, V, d, t, alpha, trainInput);
    disp(sentTree.tostring);
    %pause;

    %backpropagate
    [ dW ] = backProp( sentTree, meanings, t, outputItr, innerItr, inputItr, U, W, V, d, alpha, trainInput );

    %Numerical Differentiaton
    E=1e-6;
    [ numDiffW ] = numDiff( outputItr, innerItr, sentTree, W, U, V, d, t, alpha, E, lambda );

    %Check derivatives
    D = sum(sum((numDiffW(1:end-1) - dW(1:end-1)).^2));
    distMat=sum((numDiffW(1:end-1) - dW(1:end-1))).^2;
    
else
    i=i+1;
end

totalTime = toc(totTic);
fprintf('Checking one tree took %f seconds (aka %f minutes).\n\n',totalTime,totalTime/60);
fprintf('The squared distance matrix is: %f\n The Euclidean distance is: %f\n', distMat,D);