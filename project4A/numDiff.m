function [ numDiffW ] = numDiff( outputItr, innerItr, sentTree, W, U, V, d, t, alpha, E )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

numDiffW = zeros(d,2*d+1);
numDiffW(:,2*d+1)=W(:,2*d+1);



for i=1:d
    for j=1:2*d
        WplusE=W;
        WplusE(i,j)=W(i,j)+E;
        [ numDiffTreeWplusE ] = fwdPropNumDiff( outputItr, innerItr, sentTree, WplusE, U, V, d, t );
        [ totErrWplusE ] = totalError( outputItr, innerItr, alpha, numDiffTreeWplusE );

        WminusE=W;
        WminusE(i,j)=W(i,j)-E;
        [ numDiffTreeWminusE ] = fwdPropNumDiff( outputItr, innerItr, sentTree, WminusE, U, V, d, t );
        [ totErrWminusE ] = totalError( outputItr, innerItr, alpha, numDiffTreeWminusE );

        newDeltaW=(totErrWplusE-totErrWminusE) / (2*E);
        numDiffW(i,j)=newDeltaW;
    end
end

            
        

end

