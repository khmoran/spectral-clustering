
% the data sets
matrix1f = 'data/hamming-truth.csv';
matrix2f = 'data/hamming-test.csv';

matrix1 = csvread(matrix1f,1);
matrix2 = csvread(matrix2f,1);

hamming_dist = @(a,b)sum(a(:)~=b(:));
incorrectEntries = hamming_dist(matrix1, matrix2);

totalEntries = size(matrix1, 1)*size(matrix1, 2);

disp( incorrectEntries/totalEntries );