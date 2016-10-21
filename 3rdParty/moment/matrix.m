function X = matrix(x,n)


if nargin < 2
    n = floor(sqrt(length(x)));
    if (n*n) ~= length(x)
        error('Argument X has to be a square matrix')
    end
end
X = reshape(x,n,n);