function sub = change_to_sub(siz,ndx)
% modified from Matlab function ind2sub()

siz = double(siz);
n = length(siz);
sub = cell(1,n);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1,
  vi = rem(ndx-1, k(i)) + 1;         
  vj = (ndx - vi)/k(i) + 1; 
  sub{i} = vj; 
  ndx = vi;     
end
sub = cell2mat(sub);