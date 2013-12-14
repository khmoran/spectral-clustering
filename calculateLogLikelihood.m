% input files
dataFile = 'data/BoW-bigrams.csv';
pcaVersion = 'data/BoW-bigrams-pca.csv';
constraintsFile = 'data/aaai-constraints-mean.csv';
constraintWeightsFile = 'data/aaai-constraint-weights-skew.csv';

fprintf( '\nClustering file %s\n', dataFile );
fprintf( '\tUsing constraints %s\n', constraintsFile );
fprintf( '\tUsing constraint weights %s\n\n', constraintWeightsFile );

% load up the input files
% the data set
data = csvread(dataFile,1);
% the constraints matrix
constraints = csvread(constraintsFile);
% the weight matrix for the constraints
constraintWeights = csvread(constraintWeightsFile);

% constraint weights
ftcWt = 1;
minLambda = 0;
maxLambda = 5;
% number of ensembles 
numEnsembles = 100;
% the number of clusters desired
k = 22;
ftnum = 5;
% invariants
numLabels = 12;

% set up the parameters
% data file should have # columns = # features + n
% the #features+1...n columns should be the class labels used to produce
% constraints
length = size(data,2);
labels = data(:, length-numLabels+1:length);
numFeatures = length - numLabels;
data = data(:,1:(size(data,2)-numLabels));
orig = data;
% set all feature weights to a default of 1
featureWeights = ones(size(data,2),1);

fprintf( '\t# clusters: %i\n', k );
fprintf( '\t# features: %i\n', numFeatures );
fprintf( '\t# labels: %i\n', numLabels );
fprintf( '\t# papers: %i\n', size(data,1) );
fprintf( '\t# lambda: %i to %i\n', minLambda, maxLambda );

% k-means baseline
% Result = [];  
% [kmeansIdx,C,sumd] = kmeans(data, k, 'distance','sqeuclid', 'replicates', 20);

% spectral clusterings
numIterations = 5;
avgLLs = zeros(2, maxLambda-minLambda);
for lambda = minLambda:maxLambda
    lls = zeros(1, numIterations);

    fprintf( '\n\tlambda = %i', lambda );
    for i = 1:numIterations
        fprintf( '\n\t\ti = %i', i );
        [simMat] = generateSimilarityMatrix(data, labels, featureWeights, constraints, constraintWeights, ftcWt, lambda, ftnum, numEnsembles, k);
        [~, clusterAssignments] = spectralClustering(simMat, k);
        logLikelihood = clusteringLogLikelihood( orig, clusterAssignments, k );
        lls(1, i) = logLikelihood;
        clear clusterAssignments simMat logLikelihood;
    end

    avgLLs(1, lambda+1) = lambda;
    avgLLs(2, lambda+1) = sum(lls(1, :))/numIterations;
    
    fprintf( '\n\n\tavg log likelihood for lambda = %i: %d\n', lambda, avgLLs(2, lambda+1) );
end

plot(avgLLs(1,:), avgLLs(2,:)), xlabel('Lambda'), ylabel('Log Likelihood'), title('Log Likelihood for Various Lambda Settings');

% write out the clusters
%resultFilename = strcat('results/', strrep(strrep(dataFile, '.csv', ''), 'data/', ''), '-results.csv');
%csvwrite(resultFilename, clusterAssignments);

% determine cluster sizes using bayesian information criterion
%minK = 10;
%maxK = 20;
%data = csvread(pcaVersion,1);
%data = data(:,1:(size(data,2)-numLabels));
%bics = bicKmeans(data, minK, maxK);
