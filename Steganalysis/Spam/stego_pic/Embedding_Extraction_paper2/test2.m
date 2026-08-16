clc;
clear all;
close all;
img=imread("lena.jpg");
L=size(img);
%l=L(1)*L(2);
l=12;
%img_vec=img(:);
img_vec=[105,101,97,94;95,97,97,97;97,101,103,102];
%%% n lenght of the group/ z chane of each pixel / k the number of pixel
%%% can be change/ c array is coffisient
%
n=4;
z=1;
k=4;
C=[1;3;9;27];
ls=round(l/n);
rem_ls=mod(l,n);

%%% g is mode of 

state=state_cal(n,z,k);

%% s secret message
s=[0;65;19];

%{
excelFile = 'sec.xlsx'; 
s = xlsread(excelFile);
if isempty(s)
    fprintf('No data found in the Excel file.\n');
    return;
end
disp('Data read from Excel:');
%}
%% EXtraction orginal
%
Ext1=[];
count=1;
for i=1:n:l
    tt= double(img_vec(i:i+3));


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
%for jj=1:1:ls
for jj=1:1:ls
 det(jj)=mod((s(jj)-Ext1(jj)),state);
 chg=lookupT414(det(jj));


 img_stego(count1)=img_vec(count1)+chg(1);
 count1=count1+1;
 img_stego(count1)=img_vec(count1)+chg(2);
 count1=count1+1;
 img_stego(count1)=img_vec(count1)+chg(3);
 count1=count1+1;
 img_stego(count1)=img_vec(count1)+chg(4);
 count1=count1+1;
 
end

%img_stego=uint8(img_stego(:));
%disp(img_stego(:)-img_vec(:))
%}
%Stego = reshape(img_stego, 256, 256);
%% Histogram
%{
figure;
Histogram(img_vec,img_stego);


figure;
subplot(1, 2, 1);
imshow(img);
title('ORG');  
subplot(1, 2, 2);
imshow(Stego);
title('STEGO');

ps=psnr(img,Stego);
ss=ssim(img,Stego);
%}
%% Extraction stego

Ext2=[];
count2=1;
%for i=1:n:l-rem_ls
for i=1:n:l
    %tt1= double(img_stego(i:i+(n-1)));
    tt1= double(img_stego(i:i+3));
    Ext2(count2)=Extraction (tt1,C,n,state);
    count2=count2+1;
end

sum_Ext=sum(Ext2(:)-s(:));
%}
%{
Ext2(:);
for i=1:1:16384
if s(i)~=Ext2(i)
%disp(abs(s(i)-Ext1(i)));
disp(i);
end
end
%}
