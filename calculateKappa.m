% each cell of the file contains people's rescaled constraint values (7 ea)
filename = 'data/nominal-constraint-values.csv';
fid = fopen(filename, 'r');   % open the file
data = textscan(fid, repmat('%s', 1, 12), 'delimiter', ',', 'CollectOutput', true);
data = data{1};
fclose(fid);
results = zeros(size(data));

% calculate the fleiss kappa agreement for each cell
for r = 1:size(data, 1)
    for c = 1:size(data, 2)
         if c >= r
            cell = char(data(r, c));
            cell = strsplit(cell,'/');
            cell = str2double(cell);

            fleiss(cell)
         end
    end
end