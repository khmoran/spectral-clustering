function [clusterCenters, clusterAssignments, eigenvalues, val, U]  = spectralClustering(simMat, numClusters)
N = size(simMat,1);
D = sum(simMat, 2) + (1e-10);
D = sqrt(1./D); % D^(-1/2)
% D = spdiags(D, 0, n, n);
Dmat = zeros(N,N);
for i = 1:N
    Dmat(i,i) = D(i);
end
L = Dmat * simMat * Dmat;
eigenvalues = 1; % eig(L)';
[V, val] = eigs(L, numClusters);
V = V(:, 1:numClusters);
% Normalize each row to be of unit length
sq_sum = sqrt(sum(V.*V, 2)) + 1e-20;
U = V ./ repmat(sq_sum, 1, numClusters);
%U = V;
%for i = 1:size(U,2)
%    U(:,i) =(U(:,i)-min(U(:,i))) /(max(U(:,i))-min(U(:,i)));
%end
clear sq_sum V L;
clear D S;
%save('U.mat', 'U');
[clusterAssignments, clusterCenters] = kmeans(U, numClusters, 'EmptyAction', 'singleton', 'replicates', 50);

%[clusterCenters, clusterAssignments] = lyngby_km_main( U, 'clusters', numClusters, 'type', 'mean', 'init', 'random', 'iterations', 20, 'standardization', 'range' );

%cluster_labels = kmeans(V, num_clusters);
% nmi(cluster_labels, iris(:,5))