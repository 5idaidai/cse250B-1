%square loss functions
disp('Checking Square Loss');
x=rand(20,1);
z=rand(20,1);
num=2;
denom=4;
r=checkgrad2(@costSqLoss,x,{z,num,denom});
%assert abs(r-1)<1e-10;

%log loss functions
disp('Checking Log Loss');
l = [1,0];
p = rand;
%r=checkgrad2(@costLogLoss,l,{[p,1-p]});
%assert abs(r-1)<1e-10;

r=checkgrad2(@costLogLoss,1,{0});
r=checkgrad2(@costLogLoss,0,{1});
r=checkgrad2(@costLogLoss,1,{1});
r=checkgrad2(@costLogLoss,0,{0});
r=checkgrad2(@costLogLoss,0,{rand});
r=checkgrad2(@costLogLoss,1,{rand});

%h functions
disp('Checking H functions');
r=checkgrad2(@costH,rand*10,{});


%sigmoid functions
disp('Checking Sigmoid functions');
r=checkgrad2(@costSigmoid,rand*10,{});