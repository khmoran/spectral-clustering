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
%k = 15;
minK = 10;
maxK = 25;
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

fprintf( '\t# clusters: %i to %i\n', minK, maxK );
fprintf( '\t# features: %i\n', numFeatures );
fprintf( '\t# labels: %i\n', numLabels );
fprintf( '\t# papers: %i\n', size(data,1) );
fprintf( '\t# lambda: %i\n', lambda );

% k-means baseline
% Result = [];  
% [kmeansIdx,C,sumd] = kmeans(data, k, 'distance','sqeuclid', 'replicates', 20);

% spectral clusterings
numIterations = 2;
lls = zeros(2, maxK-minK);
bicsMax = zeros(2,maxK-minK);
bicsMin = zeros(2, maxK-minK);
for k = minK:maxK
    llsK = zeros(1, numIterations);
    bicsMaxK = zeros(1, numIterations);
    bicsMinK = zeros(1, numIterations);

    fprintf( '\n\tk = %i', k );
    for i = 1:numIterations
        fprintf( '\n\t\ti = %i', i );
        [simMat] = generateSimilarityMatrix(data, labels, featureWeights, constraints, constraintWeights, ftcWt, lambda, ftnum, numEnsembles, k);
        [~, clusterAssignments] = spectralClustering(simMat, k);
        logLikelihood = logLikelihood( orig, clusterAssignments, k );
        llsK(1, i) = logLikelihood;
        bicsMaxK(1, i) = logLikelihood - .5 * k * size(data,2) * log(size(data,1));
        bicsMinK(1, i) = -2 * logLikelihood + k * size(data,2) * log(size(data,1));
        clear clusterAssignments logLikelihood; % simMat;
    end

    lls(1, k) = k;
    lls(2, k) = sum(llsK(1, :))/numIterations;
    bicsMax(1, k) = k;
    bicsMax(2, k) = sum(bicsMaxK(1, :))/numIterations;
    bicsMin(1, k) = k;
    bicsMin(2, k) = sum(bicsMinK(1, :))/numIterations;
    
    fprintf( '\n\n\tavg log likelihood for k = %i: %d\n', k, lls(2, k) );
    fprintf( '\tavg max BIC for k = %i: %d\n', k, bicsMax(2, k) );
    fprintf( '\tavg min BIC for k = %i: %d\n', k, bicsMin(2, k) );
end

% write out the clusters
%resultFilename = strcat('results/', strrep(strrep(dataFile, '.csv', ''), 'data/', ''), '-results.csv');
%csvwrite(resultFilename, clusterAssignments);

% determine cluster sizes using bayesian information criterion
%minK = 10;
%maxK = 20;
%data = csvread(pcaVersion,1);
%data = data(:,1:(size(data,2)-numLabels));
%bics = bicKmeans(data, minK, maxK);
