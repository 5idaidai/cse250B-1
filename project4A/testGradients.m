
x=rand(20,1);
z=rand(20,1);
num=2;
denom=4;
checkgrad2(@costSqLoss,x,{z,num,denom});