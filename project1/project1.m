% Alexander Xydes, Sandi Calhoun
% CSE 250B
% Project 1

alg = str2double(inputdlg('Choose SGD (1) or L-BFGS (2)'));

if (alg == 1)
    sgd();
elseif (alg == 2)
    lbfgs();
else
    fprintf('Invalid option: %d\n', alg);
end