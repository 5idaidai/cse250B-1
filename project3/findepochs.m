prompt = 'Enter 1 to use the Classic400 dataset \nEnter 2 to use the 20Newsgroups dataset \n';
dataset = input(prompt);

if dataset==1
    file='classic400.mat';
    load(file);
    bag=classic400;
    voc=classicwordlist;
else if dataset==2
        file='20NewsgroupsShort.mat';
        load(file);
        bag=feaShort;
        voc=vocabShort;
    else if dataset>2
            print 'Not a valid dataset';
            break
        end
    end
end

fprintf('Running epoch num search on: %s\n',file);

numEpochList = [100,200,300,500,1000];
numTopics = 20;
percCutOff = 0.1;

nT = size(numEpochList,1);
times = zeros(nT);
storedThetas = cell(nT);
storedPhis = cell(nT);
storedRatios = cell(nT);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
fprintf('\n');

pidx = 1;
for numEpochs=numEpochList
    fprintf('Running LDA with %d topics, %.04f percent cutoff\n',numTopics,percCutOff);

    tic;
    [thetas,phis,ratios] = lda(bag, voc, words, numTopics, numEpochs, percCutOff);
    times(pidx) = toc;

    storedThetas(pidx) = {thetas};
    storedPhis(pidx) = {phis};
    storedRatios(pidx) = {ratios};

    fprintf('LDA took %f seconds (aka %f minutes).\n\n',times(pidx),times(pidx)/60);
    pidx = pidx + 1;
end

resultsFile = strcat(file,'_epoch_results.mat');
save(resultsFile,'times','storedThetas','storedPhis','storedRatios','voc','numTopics','percs');
