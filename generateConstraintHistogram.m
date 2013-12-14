% each cell of the file contains people's rescaled constraint values (7 ea)
filename = 'data/original-constraint-values.csv';
fid = fopen(filename, 'r');   % open the file
data = textscan(fid, repmat('%s', 1, 12), 'delimiter', ',', 'CollectOutput', true);
data = data{1};
fclose(fid);
results = zeros(size(data));

% create the plots

xvals = [-1, -.5, 0, .5, 1];
for r = 1:size(data, 1)
    for c = 1:size(data, 2)
         if c >= r
             
            % plot the BIC
            pos = size(data,1)*r + c - size(data,1);
            
            disp( pos)

            subplot(size(data, 1), size(data, 2), pos);

            cell = char(data(r, c));
            cell = strsplit(cell,';');
            cell = str2double(cell).';

            hist(cell, xvals, yvals);
            axis( [ -1 1 0 7 ] );
            set(gca,'XTick',[])
            set(gca,'YTick',[])

            % title the edges of the graph
            if c == 12
                h = text(2, 4, num2str(r) );
                set(h, 'rotation', 270)
            end
            
            if r == 1
               title(num2str(c)); 
            end
            
         end
    end
end