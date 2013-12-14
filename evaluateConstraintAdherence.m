function [ adherence ] = evaluateConstraintAdherence( labels, k, clusterAssignments, constraints, constraintWeights )
% Evaluate the clusters' adherence to the experts' constraints.
numInstances = size(labels, 1);
numLabels = size(labels, 2);

% compute the pairwise constraints values between instances
pairwiseConstraints = zeros(numInstances, numInstances);
for i = 1:numInstances
    for j = i:numInstances
        for iclass = 1:size(labels,2)
            for jclass = 1:size(labels,2)
                pairwiseConstraints(i,j) = pairwiseConstraints(i,j) + labels(i,iclass) * labels(j,jclass) * constraints(iclass, jclass) * constraintWeights(iclass, jclass);
            end
        end
    end
end

% make the matrix symmetric
for i = 1:numInstances
    for j=1:i
        pairwiseConstraints(i,j) = pairwiseConstraints(j,i);
    end
end

% calculate the total cluster constraints for each cluster by adding the
% pairwise constraints within that cluster
clusterConstraints = zeros(k, 1);
numConstraintsApplied = zeros(k, 1);
for i = 1:numInstances
    for j = 1:numInstances
        if clusterAssignments(i) == clusterAssignments(j)
            cluster = clusterAssignments(i);
            % these two papers are in the same cluster, so add their
            % constraints to that cluster's adherence
            clusterConstraints(cluster, 1) = clusterConstraints(cluster, 1) + pairwiseConstraints(i,j);
            numConstraintsApplied(cluster, 1) = numConstraintsApplied(cluster, 1) + 1;
        end
    end
end

% divide the total cluster constraints by the number of constraints applied
% to get the average constraint penalty
for c = 1:k
    if numConstraintsApplied(c, 1) > 0
        clusterConstraints(c, 1) = clusterConstraints(c, 1)/numConstraintsApplied(c, 1);
    end
end

adherence = sum(clusterConstraints(:, 1));

clear i j l c;
