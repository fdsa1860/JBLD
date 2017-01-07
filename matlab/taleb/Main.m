
%Nov-4th-2016
%Author: Taleb Alashkar
%Main function for 
% 1. pre-process the keleton data
% 2. represent that data as trajector over Grassmann manifold
% and obtain the velocity vector features 

%clean up
clear all
close all
clc



%%IN this file, we will extract the first order dervetive of skeleton
%%video after representing it as trajectory over G(m,n) manifold
addpath('/Users/Taleb/Documents/actionLoc-master/matlab/'); %path for data
%read data information
data=importdata('Data.mat');
label=importdata('Labels.mat'); 

%pre-processing
%In this step we will do translation cancelation (according to the first frame of each video
%& scal cancelation 

[preprocessed_data]= Skeleton_Preprocessing(data); 

%Trajectory analysis
options.window_size=30; 
options.remove_mean=0; 
options.step=1; 
options.PC_No=3; 
options.n=60;
options.k=3; 
options.t=0.01; 

%compute features 
VelFeaturesData=VelocityVectorFeature(preprocessed_data,options); 

%Read the feature that we computed

VelFeaturesData=importdata('VelocityVector.mat');
video_index=1; % in the dataset
Action_start_index=1;  % in the labels adjusted to our computed features
Action_end_index=100;  % in the labels adjusted to our computed features 
%compute the histogram that represent that action
Histogram=getHistogram(VelFeaturesData{video_index}(:,Action_start_index:Action_end_index)); 





