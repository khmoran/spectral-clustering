function [ logLikelihood, clusterWordCounts ] = logLikelihood( data, clusterAssignments, k )
% Calculate the log likelihood of the data given the clustering.

clusterProbs = zeros(1, k);
for c = 1:k
   numInst = sum(clusterAssignments==c);
   clusterProbs(1, c) = numInst/size(clusterAssignments, 1);
end

clusterWordCounts = zeros(k, size(data, 2));
clusterLLs = zeros(1, k);
for c = 1:k
    % get total counts for each word in cluster c
    wordCounts = zeros(1, size(data, 2));
    for fv = 1:size(data, 1)
        if clusterAssignments(fv) == c
            for word = 1:size(data, 2)
                wordCounts(1, word) = wordCounts(1, word) + data(fv, word);
            end
        end
    end
    wordCounts = wordCounts+1;

    % calculate the log likelihood for each document in cluster c
    docLLs = zeros( 1, size(data, 1 ) );
    for fv = 1:size(data, 1)
        if clusterAssignments(fv) == c
            for word = 1:size(data, 2)
                % for each word in the doc, calculate its theta hat
                if data(fv, word) > 0
                    wordProb = (wordCounts(word) + 1) / sum(wordCounts);
                    % take the log of the word probability + the log of the
                    % cluster probability
                    wordLL = log(wordProb);
                    % add this to the document total
                    docLLs(1, fv) = docLLs(1, fv) + wordLL;
                end
            end
        end
        docLLs(1, fv) = docLLs(1, fv) + log( clusterProbs( c ) );
    end
    clusterWordCounts(c, :) = wordCounts;
    clusterLLs(c) = sum(docLLs);
end

logLikelihood = sum(clusterLLs);

end