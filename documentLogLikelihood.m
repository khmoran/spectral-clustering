function [ heldOutLLs ] = documentLogLikelihood( clusterWordCounts, data, k, heldOutIdx )
% Calculate the log likelihood of the document given the induced multinomial.

% now calculate the doc log likelihoods for the held out data
heldOutLLs = zeros( 1, k );
for c = 1:k
    wordCounts = clusterWordCounts(c, :);
    for word = 1:size(data, 2)
        % for each word in the doc, calculate its theta hat
        if data(heldOutIdx, word) > 0
            wordProb = (wordCounts(word) + 1) / sum(wordCounts);
            % take the log of the word probability + the log of the
            % cluster probability
            wordLL = log(wordProb);
            % add this to the document total
            heldOutLLs(1, c) = heldOutLLs(1, c) + wordLL;
        end
    end
end

end