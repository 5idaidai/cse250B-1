function [ newTopic ] = innergibbs(i, word, topic, m, alphas, betas, q, oldn, numTopics)
%GIBBS Equation 5 in the notes
%   drawing a random number uniformly between 0 and
% 1, and using it to index into the unit interval which is divided into subintervals
% of length p

    n=oldn;
    
    n(m,topic) = n(m,topic) - 1;

    left = q(:,word) + betas(word);
    right = n(m,:)' + alphas;
    
    for j=1:numTopics
        denomLeft(j) = sum(q(j,:)'+betas);        
    end
    denomRight = sum(n(m,:)'+alphas);
    
    leftTerm = (left'./denomLeft);
    rightTerm = (right./denomRight);
    
    P = leftTerm'.*rightTerm;
    total=sum(P);
    P = P / total;
    
    r = rand;
    % r is index into 0-1 interval, subdivided into P(k) intervals
    
    tot = 0;
    for k=1:numTopics
        tot = tot + P(k);
        if r <= tot
            newTopic = k;
            return
        end
    end

end

