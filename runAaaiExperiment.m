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
pwcWt = 2;
% number of ensembles 
numEnsembles = 100;
% the number of clusters desired
k = 21;
ftnum = 5;
% invariants
numLabels = 12;
% number of iterations to determine cluster stability
numIterations = 10;

% set up the parameters
% data file should have # columns = # features + n
% the #features+1...n columns should be the class labels used to produce
% constraints
length = size(data,2);
labels = data(:, length-numLabels+1:length);
numFeatures = length - numLabels;
data = data(:,1:(size(data,2)-numLabels));
% set all feature weights to a default of 1
featureWeights = ones(size(data,2),1);

fprintf( '\t# clusters: %i\n', k );
fprintf( '\t# features: %i\n', numFeatures );
fprintf( '\t# labels: %i\n', numLabels );
fprintf( '\t# papers: %i\n', size(data,1) );
fprintf( '\t# lambda: %i\n', pwcWt );
fprintf( '\t# iterations (to calculate stability): %i\n\n', numIterations );

% store the results to determine cluster stability
clusterings = zeros(size(data,1), numIterations);
likelihoods = zeros(1, numIterations);
adherences = zeros(1,numIterations);
for i = 1:numIterations
    fprintf( '\titeration %i\n', i );
    % k-means baseline
    % Result = [];  
    % [kmeansIdx,C,sumd] = kmeans(data, k, 'distance','sqeuclid', 'replicates', 20);

    % spectral clusterings
    [simMat] = generateSimilarityMatrix(data, labels, featureWeights, constraints, constraintWeights, ftcWt, pwcWt, ftnum, numEnsembles, k);
    [~, clusterAssignments] = spectralClustering(simMat, k);
    clusterings(:,i) = clusterAssignments;

    % write out the clusters
    resultFilename = strcat('results/', strrep(strrep(dataFile, '.csv', ''), 'data/', ''), '-results.csv');
    csvwrite(resultFilename, clusterAssignments);

    % evaluate the clustering based on constraint adherence
    adherence = evaluateConstraintAdherence(labels, k, clusterAssignments, constraints, constraintWeights);
    adherences(1,i) = adherence;
    
    % evaluate the clustering based on log likelihoods
    likelihood = logLikelihood(data, clusterAssignments, k);
    likelihoods(1,i) = likelihood;
end

numChoices = 1;
if numIterations > 1
    numChoices = nchoosek(numIterations, 2);
end

stability = zeros(1, numChoices);
cell = 1;
for i = 1:numIterations
   for j = 1:numIterations
       if i < j
           stability(1, cell) = nmi(clusterings(:,i), clusterings(:,j));
           cell = cell + 1;
       end
   end
end

meanStability = sum(stability,2)/size(stability, 2);
meanAdherence = sum(adherences,2)/size(adherences, 2);
meanLikelihood = sum(likelihoods,2)/size(likelihoods, 2);

fprintf( '\n\tmean stability: %d', meanStability );
fprintf( '\n\tmean constraint adherence: %d\n', meanAdherence );
fprintf( '\n\tmean log likelihood: %d\n', meanLikelihood );

% determine cluster sizes using bayesian information criterion
%minK = 10;
%maxK = 20;
%data = csvread(pcaVersion,1);
%data = data(:,1:(size(data,2)-numLabels));
%bics = bicKmeans(data, minK, maxK);
