function AiRj = getBasis(mvect,var)
% get the subset basis from mvect basis involved with elements in var

AiRj = mvect(sum(mvect(:,~logical(sum(var,1))),2)==0,:);