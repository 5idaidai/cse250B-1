prompt = 'Enter 1 to use the Classic400 dataset \nEnter 2 to use the 20Newsgroups dataset \n';
dataset = input(prompt);

if dataset==1
    file='classic400';
    load(file);
    bag=classic400;
    voc=classicwordlist;
else if dataset==2
        file='20NewsgroupsShort';
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

numEpochs = 200;
numTopics = [2,3,4,5,10];
percCutOff = 0.1;

posbetas = [0.1, 0.05, 0.01];
posalphas = [50/numTopics, 25/numTopics, 10/numTopics, 5/numTopics];

nT = size(numTopics,1);
nB = size(posbetas,1);
nA = size(posalphas,1);

times = zeros(nA,nB,nT);
storedThetas = cell(nA,nB,nT);
storedPhis = cell(nA,nB,nT);
storedRatios = cell(nA,nB,nT);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
fprintf('\n');

kidx = 1;
aidx = 1;
bidx = 1;
for alpha=posalphas
    for beta=posbetas
        for k=numTopics
            fprintf('Running LDA with %f alpha, %f beta\n',alpha,beta);

            tic;
            [thetas,phis,ratios,alphas,betas] = lda(bag, voc, words, alpha, beta, numTopics, numEpochs, percCutOff);
            times(aidx,bidx,kidx) = toc;

            storedThetas(aidx,bidx,kidx) = {thetas};
            storedPhis(aidx,bidx,kidx) = {phis};
            storedRatios(aidx,bidx,kidx) = {ratios};

            fprintf('LDA took %f seconds (aka %f minutes).\n\n',times(aidx,bidx,kidx),times(aidx,bidx,kidx)/60);

            kidx = kidx + 1;
        end
        bidx = bidx + 1;
    end
    aidx = aidx + 1;
end

resultsFile = sprintf('%s_%dtopics_gridsearch_results.mat',file,numTopics);
save(resultsFile,'times','storedThetas','storedPhis','storedRatios','posalphas','posbetas','voc','numEpochs','numTopics','percCutOff');
%end