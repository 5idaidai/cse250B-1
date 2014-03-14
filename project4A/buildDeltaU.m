function [ deltaU ] = buildDeltaU( z, dLdGammaL, dLdGammaR )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    deltaUL = dLdGammaL(1:end-1)*z';
    deltaUR = dLdGammaR(1:end-1)*z';
    deltaC = [dLdGammaL(1:end-1); dLdGammaR(1:end-1)];
    deltaU = [deltaUL; deltaUR];
    deltaU = [deltaU, deltaC];

end

