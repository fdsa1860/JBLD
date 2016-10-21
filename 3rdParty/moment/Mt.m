function m = Mt(zz,z,index,mr)
% find the position of element in z corresponding the total dictionary zz
% and position index


if isempty(z)
    m = 1;
else
    [~,nz] = size(zz);
    if size(z,2) ~= nz
        error('z is in wrong size');
    end
    n = size(z,1);
    m = zeros(1,n);
    z(:,1:mr) = z(:,1:mr)>=1;
    for i = 1:n
%         temp = repmat(z(i,:),mz,1);
%         equ = temp==zz;
        equ = bsxfun(@eq,zz,z(i,:));
        ind = sum(equ,2)==nz;
        m(i) = index(ind);
    end
end