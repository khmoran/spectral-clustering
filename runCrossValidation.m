% Run k-fold cross-validation to produce (upper bounds of) average log
% likelihoods for various numbers of clusters.

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
minK = 15;
maxK = 22;
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

numIterations = 1;
avgLLs = zeros(2, maxK-minK);
for k = minK:maxK
    lls = zeros(1, numIterations);

    fprintf( '\n\tk = %i', k );
    for i = 1:numIterations
        fprintf( '\n\t\ti = %i', i );
        avgLL = crossValidate( numFolds, data, labels,...
            featureWeights, constraints, constraintWeights, ftcWt,...
            lambda, ftnum, numEnsembles, k );
        lls(1, i) = avgLL;
        clear avgLL;
    end

    idx = k-minK+1;
    avgLLs(1, idx) = k;
    avgLLs(2, idx) = sum(lls(1, :))/numIterations;

    fprintf( '\n\n\tavg log likelihood for k = %i: %d', k, avgLLs(2, idx) );
    clear lls;
end

plot(avgLLs(1,:), avgLLs(2,:)), xlabel('# Clusters'),...
    ylabel('Log Likelihood'), title('Log Likelihood Bound on Held Out Data');