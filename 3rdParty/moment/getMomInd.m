function M = getMomInd(ZZ,z,s,index,mr)

% M = getMomInd(ZZ,z,s,index,mr)
% organize index for moment matrix according to the total dictionary ZZ and
% index.
% z denotes the basis of moment matrix.
% s give the variable need to add on the moment matrix
% mr denotes the number of 0-1 variables

if s == 0
    s = zeros(1,size(z,2));
end
[~,n] = size(ZZ);
if n ~= size(z,2)
    error('z is in wrong size');
end
if n ~= size(s,2)
    error('s is in wrong size');
end

m = size(z,1);
temp = bsxfun(@plus,kron(ones(m,1),z)+kron(z,ones(m,1)),s);
M = reshape(getVecInd(ZZ,temp,index,mr),m,[]);
if ~issymmetric(M)
    error('Moment matrix index is not symmetric!');
end
% M = -1*ones(size(z,1),size(z,1));
% parfor i = 1:size(z,1)
%     temp = bsxfun(@plus,z,z(i,:)+s);
%     M(i,:) = getVecInd(ZZ,temp,index,mr);
% end