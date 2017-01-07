function [PC,V,mn] = pca2(data,Remove_Mean)
% PCA2: Perform PCA using SVD.
% data - MxN matrix of input data
% (M dimensions, N trials)
% signals - MxN matrix of projected data
% PC - each column is a PC
% V - Mx1 matrix of variances
[M,N] = size(data);
% subtract off the mean for each dimension
mn = mean(data,2);
%figure(3);title('Mean'); 

%imshow(mn)
if Remove_Mean
data = data - repmat(mn,1,N);
%construct the matrix Y
Y = data'/ sqrt(N-1);
else
  Y = data'; 
end 

% SVD does it all
[u,S,PC] = svd(Y,'econ');
% calculate the variances
S = diag(S);
V = S .* S;

% project the original data
%signals = PC' * data;

