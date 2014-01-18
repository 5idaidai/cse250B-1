% Alexander Xydes, Sandi Calhoun
% CSE 250B
% Project 1

w=0.5;
mu=0.01;
x=rand(1000,1);
y=rand(1000,1)>0.5;
checkgrad2(@costFunction,w,{x,y,mu});

initial_theta=w;
    options = [];
    options.Method = 'lbfgs';
[x,f,exitflag,output] = ...
    minFunc(@costFunction,...
    initial_theta,options,{w,x,y,mu})


    %minFunc(@(t)(costFunction(t, x, y, mu)),...

pause;

alg = str2double(inputdlg('Choose SGD (1) or L-BFGS (2)'));

if (alg == 1)
    sgd();
elseif (alg == 2)
    lbfgs();
else
    fprintf('Invalid option: %d\n', alg);
end