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
lambda = 2;
% number of ensembles 
numEnsembles = 100;
% the number of clusters desired
minK = 10;
maxK = 25;
ftnum = 5;
% invariants
numLabels = 12;
numFolds = 5;%10;

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

fprintf( '\t# clusters: %i to %i\n', minK, maxK );
fprintf( '\t# features: %i\n', numFeatures );
fprintf( '\t# labels: %i\n', numLabels );
fprintf( '\t# papers: %i\n', size(data,1) );
fprintf( '\t# lambda: %i\n', lambda );
fprintf( '\t# folds: %i\n', numFolds );

% k-means baseline
% Result = [];  
% [kmeansIdx,C,sumd] = kmeans(data, k, 'distance','sqeuclid', 'replicates', 20);

% spectral clusterings
numIterations = 5; %10;
avgSils = zeros(2, maxK-minK);
for k = minK:maxK
    sils = zeros(1, numIterations);

    fprintf( '\n\tk = %i', k );
    for i = 1:numIterations
        fprintf( '\n\t\ti = %i', i );
        [simMat] = generateSimilarityMatrix(train, labs, featureWeights,...
            constraints, constraintWeights, ftcWt, lambda, ftnum, numEnsembles, k);
        [~, clusterAssignments] = spectralClustering(simMat, k);
        s = silhouette(orig, clusterAssignments);
        clear clusterAssignments simMat logLikelihood;
    end

    avgSils(1, k-minK+1) = k;
    avgSils(2, k-minK+1) = sum(sils(1, :))/numIterations;
    
    fprintf( '\n\n\tavg silhouette for k = %i: %d\n', k, avgSils(2, k+1) );
end

plot(avgSils(1,:), avgSils(2,:)), xlabel('# Clusters'),...
    ylabel('Log Likelihood'), title('Log Likelihood on Held Out Data');

% write out the clusters
%resultFilename = strcat('results/', strrep(strrep(dataFile, '.csv', ''), 'data/', ''), '-results.csv');
%csvwrite(resultFilename, clusterAssignments);

% determine cluster sizes using bayesian information criterion
%minK = 10;
%maxK = 20;
%data = csvread(pcaVersion,1);
%data = data(:,1:(size(data,2)-numLabels));
%bics = bicKmeans(data, minK, maxK);
