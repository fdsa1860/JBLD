function solution = extract_solution(moment,r,mvect,n)
% moment: moment matrix
% r: moment matrix rank
% mvect: power list of moment matrix basis
% n: number of binary variables

[U,S,V] = svd(moment);
s = diag(S);
cholesky = zeros(size(U));
for i = 1:size(U,2)
    cholesky(:,i) = U(:,i)*sqrt(s(i));
end
if norm(cholesky*cholesky'-U*S*V','fro') > 1e-8*numel(moment)
    error('Factor Problem!');
end
E = rref(cholesky(:,1:r)')';
E(abs(E)<=1e-8) = 0;
[x,y] = find(E~=0);
[Tx,Ty] = size(mvect);
ind = zeros(Tx,1);
for i = 1:r
    temp = x(y==i);
    ind(temp(1)) = 1;
end
w = mvect(logical(ind),:);
var = eye(Ty);
lambda = normr(rand(1,Ty));
N = zeros(size(w,1),r);
Ni = cell(1,Ty);
for i = 1:Ty
    temp = repmat(var(i,:),size(w,1),1)+w;
    ind_temp = getVecInd(mvect,temp,1:Tx,n);
    Ni{i} = E(ind_temp,:);
    N = N+lambda(i)*Ni{i};
end
[Q,T] = schur(N);
solution = zeros(Ty,r);
for i = 1:Ty
    for j = 1:r
        solution(i,j) = Q(:,j)'*Ni{i}*Q(:,j);
    end
end