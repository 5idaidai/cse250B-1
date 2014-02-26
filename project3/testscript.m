
prompt = 'Enter 1 to use the Classic400 dataset \nEnter 2 to use the 20Newsgroups dataset \n';
dataset = input(prompt);

if dataset==1
    load('classic400.mat');
    bag=classic400;
    voc=classicwordlist;
    labels=truelabels;
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

prompt = 'Enter the alpha value (which will be divided by numTopics) \n';
alpha = input(prompt)/numTopics;

prompt = 'Enter the beta value \n';
beta = input(prompt);

words = get_words(bag, voc, sum(sum(bag)), size(bag,1));
    
tic;
[thetas,phis,ratios,alphas,betas,change] = lda(bag, voc, words, alpha, beta, numTopics, numEpochs, percCutOff);
TimeSpent = toc;

fprintf('LDA took %f seconds (aka %f minutes).\n\n',TimeSpent,TimeSpent/60);

printTopKWords(phis,voc,20);

if dataset==1
    figure(1);
    hold on;
    plotDocTopics(thetas);
    plotTrueLabels(labels);
    grid on;
    xlabel('Topic 1','fontsize',14,'fontweight','bold');
    ylabel('Topic 2','fontsize',14,'fontweight','bold');
    zlabel('Topic 3','fontsize',14,'fontweight','bold');
    title('Classic400 Dataset Theta Distribution','fontsize',16,'fontweight','bold');
    hold off;
    saveas(gcf,'Classic400.fig');
    save('c400Data.mat','ratios','thetas','phis','alphas','betas','percCutOff','voc','change');

else if dataset==2
    figure(2);
    %hold on;
    plotDocTopics(thetas);
    %plotTrueLabels(labels);
    grid on;
    xlabel('Topic 1','fontsize',14,'fontweight','bold');
    ylabel('Topic 2','fontsize',14,'fontweight','bold');
    zlabel('Topic 3','fontsize',14,'fontweight','bold');
    title('20Newsgroups Dataset Theta Distribution','fontsize',16,'fontweight','bold');
    %hold off;
    saveas(gcf,'20Newsgroups.fig');
    save('20nData.mat','ratios','thetas','phis','alphas','betas','percCutOff','voc','change');
    end
end
