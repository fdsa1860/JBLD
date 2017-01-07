clear all
close all
clc


Sub_No=3; 
Expresssion='Happy'; 
PC_No=6; 
Remove_Mean=0; 


path=strcat('D:\vrml2depth\vrml2depth\chemin_geodesic\Sub',num2str(Sub_No),'_',Expresssion,'_mc_'); 
S=[];    % img matrix

M=50; %Number of Frames 

for i=1:1:M
    filename=strcat(path,int2str(i),'.png')
    I1=imread(filename); 
    I2=squeeze(I1(:,:,1));
    I3=imcrop(I2,[70 40 160 220]); 

     I3=imresize(I3,0.75); 
    [irow icol]=size(I3) ;

    imshow(I3)
    title1=strcat('Frame: ',num2str(i)); 
    title(title1);
    axis equal
    colormap hot
    pause(0.1)
    
    temp=reshape(I3',irow*icol,1);    % creates a (N1*N2)x1 vector
    S=[S temp];    % S is a N1*N2xM matrix after finishing the sequence

end 

% Change image for manipulation
dbx=[];    % A matrix
for i=1:M
temp=double(S(:,i));
dbx=[dbx temp];
end

data=dbx; 

[PC,V,Mean] = pca2(data,Remove_Mean);
delete('Test.txt'); 
dlmwrite('Test.txt',PC,'delimiter',' '); 

tmimg=uint8(Mean); % converts to unsigned 8-bit integer. Values range from 0 to 255
img=reshape(tmimg,icol,irow); % takes the N1*N2x1 vector and creates a N1xN2 matrix
img=img'; 
figure(2);
imagesc(img);
axis equal
%colormap gray
title('Mean Image','fontsize',18)

figure(3);
 %title('Eigenfaces','fontsize',18)
subplot(1,PC_No+1,1); 
imagesc(img);
axis off
axis equal
title('Mean')

for i=1:PC_No    
img=reshape(PC(:,i),icol,irow);
%Mean=reshape(Mean,icol,irow); 
img=img';
%img=histeq(img,255);
subplot(1,PC_No+1,i+1); 
imagesc(img)
axis equal
axis off

%colormap  gray
title2=strcat('PC:',num2str(i)); 
title(title2)
%drawnow;


end

% 
% xx=importdata('Test.txt');
figure(10);
%icol=161;
%irow=221; 
icol2=0;
irow2=0; 

figure(12); 
for i=1:1:5
img1=reshape(PC(:,i),icol,irow);
img1=img1';
%img1=histeq(img1,255);
subplot(1,5,i); 
%figure(i)
imagesc(img1)
colormap hot

drawnow;
pause(0.2)
%if i==5
filename2=strcat('karcher Mean: PC:',num2str(i)); 
%title(filename2,'fontsize',12)
%end
end 




