function [ns] = nCalc(wordsPerDoc,z,m,k)
% Calculates the number of times topic j occurs in document m

    ns = zeros(m,k);
    zidx = 1;

    for i=1:m
        for w=1:wordsPerDoc(i)
            j = z(zidx);
            ns(i,j)=ns(i,j)+1;
            zidx = zidx+1;
        end
    end
end
        
            
        
        
    

     
