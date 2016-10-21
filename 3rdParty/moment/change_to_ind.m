function ndx = change_to_ind(siz,I)
% modified from Matlab function sub2ind()
siz = double(siz);
if length(siz)<1
        error('InvalidSize');
end

%Compute linear indices
k = [1 cumprod(siz(1:end-1))];
ndx = 1;
for i = 1:length(siz),
    v = I(:,i);
    if (any(v(:) < 1)) || (any(v(:) > siz(i)))
        %Verify subscripts are within range
        error('Index Out Of Range');
    end
    ndx = ndx + (v-1)*k(i);
end
