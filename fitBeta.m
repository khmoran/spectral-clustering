% each cell of the file contains people's rescaled constraint values (7 ea)
filename = 'data/rescaled-constraint-values.csv';
fid = fopen(filename, 'r');   % open the file
data = textscan(fid, repmat('%s', 1, 12), 'delimiter', ',', 'CollectOutput', true);
data = data{1};
fclose(fid);
results = zeros(size(data));

% fit a beta distribution for each cell
for r = 1:size(data, 1)
    for c = 1:size(data, 2)
         if c >= r
            cell = char(data(r, c));
            cell = strsplit(cell,';');
            cell = str2double(cell).';

            pd = fitdist(cell,'beta');
            mean = pd.mean;
            rescaled = (mean*2)-1;

            % save the rescaled mean of the beta
            results(r, c) = rescaled;

            % plot the distribution
            %domain = [0.0:0.01:1.0];
            %y = betapdf(domain, pd.a, pd.b);
            %figure;
            %plot(domain, y, 'Color', 'r', 'LineWidth', 2);
         end
    end
end