function [changedThetas] = checkThetas(z,wordsPerDoc,thetasOld,m,k)

    changedThetas=zeros(m,k);
    n = nCalc(wordsPerDoc,z,m,k);
    thetas = n;
    for i=1:m
        r1sum=sum(n(i,:));
        for j=1:k
            thetas(i,j) = n(i,j)/r1sum;
        end
        changedThetas(i,:)=thetas(i,:)-thetasOld(i,:);
    end

end
   
   
    

