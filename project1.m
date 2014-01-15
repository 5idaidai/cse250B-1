% Alexander Xydes, Sandi Calhoun
% CSE 250B
% Project 1

alg = str2double(inputdlg('Choose SGD (1) or L-BFGS (2)'));

if (alg == 1)
    fprintf('Running SGD on the dataset\n');
    sgd();
elseif (alg == 2)
    fprintf('Running L-BFGS on the dataset\n');
    lbfgs();
else
    fprintf('Invalid option: %d\n', alg);
end