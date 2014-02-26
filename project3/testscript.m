
prompt = 'Enter 1 to use the Classic400 dataset \nEnter 2 to use the 20Newsgroups dataset \n';
dataset = input(prompt);

if dataset==1
    load('classic400.mat');
    bag=classic400;
    voc=classicwordlist;
else if dataset==2
        load('20NewsgroupsShort.mat');
        bag=feaShort;
        voc=vocabShort;
    else if dataset>2
            print 'Not a valid dataset';
            break
        end
    end
end

prompt = 'Enter the number of epochs \n';
numEpochs = input(prompt);

prompt = 'Enter the number of topics \n';
numTopics = input(prompt);

prompt = 'Enter the percent cutoff in decimal format \n';
percCutOff = input(prompt);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
    
tic;
[thetas,phis,ratios,alphas,betas,change] = lda(bag, voc, words, numTopics, numEpochs, percCutOff);
TimeSpent = toc;

fprintf('LDA took %f seconds (aka %f minutes).\n\n',TimeSpent,TimeSpent/60);

plotDocTopics(thetas);
printTopKWords(phis,voc,10);

if dataset==1
    save('c400Data.mat','ratios','thetas','phis','alphas','betas','percCutOff','voc','change');
else if dataset==2
        save('20nData.mat','ratios','thetas','phis','alphas','betas','percCutOff','voc','change');
    end
end
