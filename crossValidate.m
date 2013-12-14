function avgLL = crossValidate( numFolds, data, labels, featureWeights, constraints, constraintWeights, ftcWt, lambda, ftnum, numEnsembles, k )
% Get an upper bound on the log likelihood of held out data.
%   This method gets an upper bound by assigning each feature vector to
%   the cluster that maximizes the log likelihood of the data.

cvo = cvpartition(size(data, 1), 'kfold', numFolds);

foldLLs = zeros(numFolds, 1);
for fold = 1:cvo.NumTestSets
    fprintf('\n\t\t\tfold = %i', fold );
    trainIdx = cvo.training(fold);
    testIdx = cvo.test(fold);
    train = data(trainIdx, :);
    test = data(testIdx, :);
    labs = labels(trainIdx, :);

    % cluster based on the training set
    [simMat] = generateSimilarityMatrix(train, labs, featureWeights,...
        constraints, constraintWeights, ftcWt, lambda, ftnum, numEnsembles, k);
    [~, clusterAssignments] = spectralClustering(simMat, k);

    % compute log likelihood of the held out data
    highestLLs = zeros(size(test, 1), 1);
    [~, wordCounts] = clusteringLogLikelihood(train, clusterAssignments, k);
    for t = 1:size(test,1)
       heldOutLLs = documentLogLikelihood(wordCounts, data, k, t);
       highestLLs(t, 1) = max(heldOutLLs);
    end

    foldLLs(fold, 1) = mean(highestLLs, 1);
end

avgLL = mean(foldLLs, 1);

end

