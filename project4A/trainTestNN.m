
%init W and b randomly
%a + (b-a).*rand(100,1)
W = rand(1,2);%[0.2,0.4];
t=0;
d=1;

%Training
%get sentence
sent=[1,1,1];

%build test tree
rootnode = cell(13,1);
rootnode{2} = 3;
sentTree = tree(rootnode);

hidden = cell(13,1);
ht = tree(hidden);
[sentTree] = sentTree.graft(1,ht);
hidx = 2;

firstinput = cell(13,1);
firstinput{1} = 1;
firstinput{2} = 2;
nt = tree(firstinput);
sentTree = sentTree.graft(1, nt);

for i=1:2
    node = cell(13,1);
    node{1} = 1;
    node{2} = 1;
    nt = tree(node);
    sentTree = sentTree.graft(2, nt);
end

disp(sentTree.tostring());

%forward prop
sentTree = forwardPropTestNN(sentTree,W,hidx,t);

%backpropagate hardcoded
dW = zeros(size(W));

%root
node = sentTree.get(1);
children = sentTree.getchildren(1);
childr = sentTree.get(max(children));
childl = sentTree.get(min(children));

dRoot = gradSquareLoss(t,node{1},1,1,1);
dWR = dRoot * [childl{1};childr{1}]';

%hidden
node = sentTree.get(hidx);
children = sentTree.getchildren(hidx);
childr = sentTree.get(max(children));
childl = sentTree.get(min(children));

parent = sentTree.getparent(hidx);
nodes = sentTree.getchildren(parent);
nodeL = min(nodes);
if nodeL==hidx
    Wk = W(:,1:d);
else
    Wk = W(:,d+1:2*d);
end

dHidden = hprime(node{1}) * (dRoot*Wk);
dWH = dHidden * [childl{1};childr{1}]';

%sum
dW = dWR + dWH;

%true deriv
root=sentTree.get(1);
a=W(1) + W(2);
dTW(1) = root{1} * (h(a) + W(1)*hprime(a));
dTW(2) = root{1} * (1 + W(2)*hprime(a));

%Numerical Differentiaton
numDiffW = zeros(size(W));
E=1e-6;
for i=1:d
    for j=1:2*d
        WplusE=W;
        WplusE(i,j)=W(i,j)+E;
        [ numDiffTreeWplusE ] = forwardPropTestNN( sentTree, WplusE, hidx, t );
        [ totErrWplusE ] = totalErrorTNN( 1, numDiffTreeWplusE, WplusE );

        WminusE=W;
        WminusE(i,j)=W(i,j)-E;
        [ numDiffTreeWminusE ] = forwardPropTestNN( sentTree, WminusE, hidx, t );
        [ totErrWminusE ] = totalErrorTNN( 1, numDiffTreeWminusE, WminusE );

        newDeltaW=(totErrWplusE-totErrWminusE) / (2*E);
        numDiffW(i,j)=newDeltaW;
    end
end

%Check derivatives
D = sum(sum((dTW - dW).^2));
D2 = sum(sum((numDiffW - dW).^2));

s=sum(sum(dW));
sn=sum(sum(numDiffW));

fprintf('The Euclidean distance is %f, %f\n', D, D2);