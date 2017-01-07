% Get the histogram of the velocity vectors for one action
% Nov-4th-2016
% Author: Taleb Alashkar

% Input: The velocity vectors that resides between the onset and the offset
% of the action as a Matrix of size m by n
% where m:57 n: number of velocity vectors

%Output: The 2*57 positive-negative histogram 



function [JointHist] = getHistogram(ActionMat)

[m n]=size(ActionMat); 

JointHist=zeros(2*m,1); 

for i=1:1:m
    for j=1:1:n
        if (ActionMat(i,j)>0)
            JointHist(2*(i-1)+1) = JointHist(i)+ActionMat(i,j);  
        else 
            JointHist(2*i) = JointHist(i)+abs(ActionMat(i,j)); 
        end 
    end 
end 

 

end 