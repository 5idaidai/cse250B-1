function [sentTree] = forwardPropTestNN(sentTree, W, hidx, t)

    %hidden node
    node = sentTree.get(hidx);
    children = sentTree.getchildren(hidx);
    childr = sentTree.get(max(children));
    childl = sentTree.get(min(children));

    xl = childl{1};
    xr = childr{1};

    nl = childl{2};
    nr = childr{2};

    a = W * [xl ; xr];
    xk = h(a);

    node{1} = xk;
    node{2} = nl + nr;

    sentTree = sentTree.set(hidx,node);

    %root node
    node = sentTree.get(1);
    children = sentTree.getchildren(1);
    childr = sentTree.get(max(children));
    childl = sentTree.get(min(children));

    xl = childl{1};
    xr = childr{1};

    nl = childl{2};
    nr = childr{2};

    a = W * [xl ; xr];
    xk = (a);

    node{1} = xk;
    node{2} = nl + nr;
    node{13} = squareLoss(t,xk,1,1,1);

    sentTree = sentTree.set(1,node);

end