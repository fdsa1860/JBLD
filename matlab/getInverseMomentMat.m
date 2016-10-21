function [Minv] = getInverseMomentMat(data, mord)

n = size(data, 1);
nVar = size(data, 2);
% mord = 4;
[Dict,Indx] = momentPowers(0, nVar, 2*mord);
[basis,~] = momentPowers(0, nVar, mord);
Mi = getMomInd(Dict,basis,0,Indx,0);

m = zeros(1, length(Indx));
for i = 1:length(m)
    m(i) = sum( prod(bsxfun(@power, data, Dict(i,:)),2) ) / n;
end

M = m(Mi);

Minv = inv(M+1e-10*eye(size(M)));

end