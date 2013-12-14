function [simMat, ensemble, labels, data] = generateSimilarityMatrix(data, labels, ftwts, constraintMatrix, constraintWeights, ftcWt, pwcWt, ftnum, numEn, k)

  nft = size(data, 2);
  npt = size(data,1);
  
  % scale all features to [0,1] interval
  for i = 1:nft
      if max(data(:,i))-min(data(:,i)) > 0
          data(:,i) = (data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
      end
  end
  
%  ftwts = csvread(ftwtFilename);
%  ftwts = ftwts;
%  ftwts = ones(1,nft)/nft;
  ftwts = ftwts ./mean(ftwts);
%  groupConMat = csvread(consMatFilename);
  
  % get cluster ensemble
  ensemble = zeros(numEn, npt);
  numftEn = numEn*ftcWt;
  for i = 1:numEn
 %     [submat, ptInd] = sub_mat(data, ftwts, ptsec, ftnum);
 %     ensem = zeros(1,npt);
      Rmat = normrnd (0,1,nft, ftnum);
      if i<=numftEn
          for rrow = 1:nft
              Rmat(rrow,:) = Rmat(rrow,:)*ftwts(rrow);
          end
      end
      Rmat = normc(Rmat);
      pdata = data*Rmat;
      idx = kmeans(pdata, k, 'EmptyAction', 'drop','replicates',20);
 %     ensem(ptInd) = idx;
      ensemble(i,:) = idx;
  end
  
  % get simMat
  simMat = zeros(npt);
  for nEn = 1:numEn
      for i = 1:npt
          for j = i:npt
              if ensemble(nEn,i) == ensemble(nEn,j)
                  simMat(i,j) = simMat(i,j) + 1;
              end
          end
      end
  end
  
  
  % compute NxN matrix of constraints
  pwConMat = zeros(npt, npt);
  for i = 1:npt
      for j = i:npt
          for iclass = 1:size(labels,2)
              for jclass = 1:size(labels,2)
                  pwConMat(i,j) = pwConMat(i,j) + labels(i,iclass) * labels(j,jclass) * constraintMatrix(iclass, jclass) * constraintWeights(iclass, jclass);
              end
          end
      end
  end
  %make a symatric matrix
  for i = 1:npt
      for j=1:i
          pwConMat(i,j) = pwConMat(j,i);
      end
  end
  
  simMat = simMat/numEn;
  %put in group constraints
  for i = 1:npt
      for j = i:npt
%          simMat(i,j) = simMat(i,j) * exp (pwcWt*groupConMat(labels(i,1),labels(j,1)));
          simMat(i,j) = 2*simMat(i,j) + pwcWt*pwConMat(i,j)+1;
      end
  end
  
  %make a symatric matrix
  for i = 1:npt
      for j=1:i
          simMat(i,j) = simMat(j,i);
      end
  end
  
  
  