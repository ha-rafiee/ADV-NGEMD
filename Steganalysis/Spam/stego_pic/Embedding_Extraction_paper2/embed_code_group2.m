clc;
clear all;
close all;
img=imread("img\baboon.tif");
L=size(img);
l=L(1)*L(2);
img_vec=img(:);
%%% n lenght of the group/ z chane of each pixel / k the number of pixel
%%% can be change/ c array is coffisient
n=2;
z=3;
k=2;
C=[1;7];
ls=floor(l/n);
rem_ls=mod(l,n);

%%% g is mode of 
state=49;

%%% s secret message
s=randi([0,state-1],1,ls);
%% Read change data from Excell
excelFile = 'data/d232.xlsx'; 
LookupT = xlsread(excelFile);
if isempty(LookupT)
    fprintf('No data found in the Excel file.\n');
    return;
end
disp('Data read from Excel:');
%% EXtraction orginal
%
Ext1=[];
count=1;
for i=1:n:l-rem_ls
%for i=1:n:4
    tt= double(img_vec(i:i+(n-1)));

    Ext1(count)=Extraction(tt,C,n,state);
    %Ext1(count)=Extraction(img_vec(i:i+(n-1)),C,n,state);
    count=count+1;
end
%}
%% Embedding
%
chg=[];
img_stego=[];
det=[];
count1=1;
for jj=1:1:ls
 det(jj)=mod((s(jj)-Ext1(jj)),state);
% chg=lookupT232(det(jj));
 for ic=1:1:state
     if( det(jj)== LookupT(ic,3))
         chg(1)=LookupT(ic,1);
         chg(2)=LookupT(ic,2);
     end
 end

 img_stego(count1)=img_vec(count1)+chg(1);
 count1=count1+1;
 img_stego(count1)=img_vec(count1)+chg(2);
 count1=count1+1;
 
end

img_stego=uint8(img_stego(:));
%disp(img_stego(:)-img_vec(:))
%}
Stego = reshape(img_stego, L(1),L(2));
%% Histogram
%

hist1 = imhist(img);
hist2 = imhist(Stego);

plot(hist1,LineWidth=1.5);
hold on;
plot(hist2,LineWidth=1.5);xlabel('Value',FontSize=11,FontWeight='bold');ylabel('Histogrm of two Image',FontSize=11,FontWeight='bold');
legend('Orginal Image','Stego Image')


figure;
subplot(1, 2, 1);
imshow(img);
title('ORG');  
subplot(1, 2, 2);
imshow(Stego);
title('STEGO');
%}
ps=psnr(img,Stego);
ss=ssim(img,Stego);
%% Extraction stego

Ext2=[];
count2=1;
for i=1:n:l-rem_ls
%for i=1:n:4
    tt1= double(img_stego(i:i+(n-1)));
    Ext2(count2)=Extraction(tt1,C,n,state);
    count2=count2+1;
end

sum_Ext=sum(Ext2(:)-s(:));
for i=1:1:32768
if(Ext2(1,i)~=s(1,i))
disp(i);
disp(Ext2(i));
disp(s(i));
end
end


%}