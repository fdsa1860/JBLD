
%Skeleton Data precocessing: Scale and Translation Cancellation 
%Nov-4th-2016
%Author: Taleb Alashkar

%This function cancel the translation for the whole skeleton video
%with respect to the very first frame and cancel the scaleing 


function [processedData] = Skeleton_Preprocessing(Data)

number_of_joints=35;
number_of_coordinates=3; 

number_of_videos=size(Data,2); 

for i=1:1:number_of_videos
    
    video_ith=Data{i}; 
    first_frame_of_video_ith = video_ith(:,1); 
    Skeleton_Matrix = reshape(first_frame_of_video_ith,number_of_coordinates,number_of_joints)'; 
    muX = mean(Skeleton_Matrix,1);
    number_of_frames = size(video_ith,2); 
    preprocessedvideo = zeros(size(video_ith,1),size(video_ith,2)); 
    
    for j=1:1:number_of_frames; 
      frame_jth=video_ith(:,j); 
      Skeleton_Matrix = reshape(frame_jth,number_of_coordinates,number_of_joints)'; 
      Trans_Skeleton_Matrix = Skeleton_Matrix-repmat(muX, number_of_joints, 1);
      
      normX = sqrt(trace(Trans_Skeleton_Matrix*Trans_Skeleton_Matrix'));
      % scale to equal (unit) norm
      Scaled_Trans_Skel_Mat = Trans_Skeleton_Matrix / normX;
      preprocessedvideo(:,j)=reshape(Scaled_Trans_Skel_Mat',1,number_of_joints*number_of_coordinates); 

    end 
    processedData{i}=preprocessedvideo; 
end 

end 