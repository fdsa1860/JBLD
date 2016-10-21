function M = Mind(ZZ,z,s,index,mr)
% organize index for moment matrix according to the total dictionary ZZ and
% index

if s == 0
    s = zeros(1,size(z,2));
end
[m,n] = size(ZZ);
if n ~= size(z,2)
    error('z is in wrong size');
end
if n ~= size(s,2)
    error('s is in wrong size');
end
    
M = [];
parfor i = 1:size(z,1)
%     temp = z+repmat(z(i,:),size(z,1),1)+repmat(s,size(z,1),1);
    temp = bsxfun(@plus,z,z(i,:)+s);
%     temp(:,1:mr) = temp(:,1:mr)>=1;
    M(i,:) = Mt(ZZ,temp,index,mr);
end