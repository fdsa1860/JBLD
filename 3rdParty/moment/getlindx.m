function [lindx,l] = getlindx(a)
% input a vector 'a', calculate the 1D coordinates in a structure
% matrix 'lindx' and row number of structure matrix
% got from Mustafa

if size(a,2)~=1
    a = a';
end
l = length(a);
lindx = (a-1)*l+(1:l)';
% [ele,loc] = sort(a);
% lindx = loc+(ele-1)*length(ele);
lindx(lindx<=0) = [];
% l = length(ele);