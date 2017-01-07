
%This function to compute the velocity vector features out of Concurrent
%Action Dataset 3D skeleton videos 
%Author: Taleb Alashkar
%Nov-4-2016



function [VelocityVector] = VelocityVectorFeature(preprocessed_data,options)


window_size=options.window_size; %sliding window size 
remove_mean=options.remove_mean; %remove mean or not for PCA
step=options.step;               %Sliding window step
PC_No=options.PC_No;             %# of princpal component to keep
n=options.n;                     % size of feature 20*3=60
k=options.k;                     % size of observation xyz=3
t=options.t;                     % geodeisc step = 0.01

for i=1:1:size(preprocessed_data,2)
   
    video_ith=preprocessed_data{i}; 
    %build subspace for every 60 successive frame no overlapping 
    number_of_frames=size(video_ith,2); 
    number_of_subspaces=floor((number_of_frames-window_size)/step); 
    
    subspace_index=1;
    
    M_ref = video_ith(:,subspace_index:subspace_index+window_size-1); 
    [PC,V,Mean] = pca2(M_ref,remove_mean);
    Sub_ref=PC(:,1:PC_No);
    
    %subspace_index=window_size+1;
    
    Vel_Vect_Traj=zeros(n-k,1); 
    
    for j=1:1:number_of_subspaces
        subspace_index=subspace_index+step; 
        M_curr = video_ith(:,subspace_index:subspace_index+window_size-1); 
        [PC,V,Mean] = pca2(M_curr,remove_mean);
        Sub_curr=PC(:,1:PC_No);
        
        B = compute_velocity_grassmann_efficient(Sub_ref,Sub_curr); 
        
        I = [eye(k);zeros(n-k,k)]; 
        
        
        %%%% Compute the velocity direction between one grassmann element of the
%%%% trajectory and the identity element

         A=compute_velocity_grassmann_efficient(Sub_ref,I); 


%%% Compute the Paralle Tranport of the velocity vector between P1,P2
         tB= Grassmann_ParallelTranslate(Sub_ref, B, A, t); 

         Vel_Vect_Traj(:,j)=tB(:,1); 
        
        stop=1; 
        
    end 
    VelocityVector{i}=Vel_Vect_Traj; 
end 

% save('VelocityVector.mat','VelocityVector'); 




end 