clc;
clear all;
close all;
img=imread("img/lena.jpg");
%img=imread("img/barbara.bmp");
L=size(img);
l=L(1)*L(2);
img_vec=img(:);
%%% n lenght of the group/ z chane of each pixel / k the number of pixel
%%% can be change/ c array is coffisient
n=4;
z=3;
k=4;
C=[1;7;49;343];
ls=floor(l/n);
rem_ls=mod(l,n);
remind=img_vec(l-rem_ls+1:l);
remind=remind';
%%% g is mode of 
%% Read change data from Excell
excelFile = 'data/d434.xlsx'; 
LookupT = xlsread(excelFile);
if isempty(LookupT)
    fprintf('No data found in the Excel file.\n');
    return;
end
disp('Data read from Excel:');
%%
szl=size(LookupT);
state=2401;

%%% s secret message
s=randi([0,state-1],1,ls);


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
     if( det(jj)== LookupT(ic,5))
         chg(1)=LookupT(ic,1);
         chg(2)=LookupT(ic,2);
         chg(3)=LookupT(ic,3);
         chg(4)=LookupT(ic,4);
     end
 end

 img_stego(count1)=img_vec(count1)+chg(1);
 count1=count1+1;
 img_stego(count1)=img_vec(count1)+chg(2);
 count1=count1+1;
 img_stego(count1)=img_vec(count1)+chg(3);
 count1=count1+1;
img_stego(count1)=img_vec(count1)+chg(4);
 count1=count1+1;
                                                                                                            
end

img_stego1=[img_stego,remind];
img_stego1=uint8(img_stego1(:));
%disp(img_stego(:)-img_vec(:))
%}
Stego = reshape(img_stego1, L(1), L(2));
%% Histogram
%
figure;
hist1 = imhist(img);
hist2 = imhist(Stego);

plot(hist1,LineWidth=1.5);
hold on;
plot(hist2,LineWidth=1.5);xlabel('Value',FontSize=11,FontWeight='bold');ylabel('Histogrm of two Image',FontSize=11,FontWeight='bold');
legend('Orginal Image','Stego Image')


%
%Histogram(img_vec,img_stego1);

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
    tt1= double(img_stego1(i:i+(n-1)));
    Ext2(count2)=Extraction(tt1,C,n,state);
    count2=count2+1;
end

sum_Ext=sum(Ext2(:)-s(:));
for i=1:1:32768
%if(Ext2(1,i)~=s(1,i))
%disp(i);
%disp(Ext2(i));
%disp(s(i));
%end
end
imwrite(Stego, "StegoImg/stego4.tif")

%}