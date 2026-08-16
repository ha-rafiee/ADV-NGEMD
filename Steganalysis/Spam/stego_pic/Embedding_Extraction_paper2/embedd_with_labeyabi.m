clc;
clear all;
close all;
%img=imread("img/lena.jpg");
img=imread("img/baboon.tif");
L=size(img);
l=L(1)*L(2);
img_vec=img(:);
%%% n lenght of the group/ z chane of each pixel / k the number of pixel
%%% can be change/ c array is coffisient
n=5;

ls=floor(l/n);
rem_ls=mod(l,n);
remind=img_vec(l-rem_ls+1:l);
remind=remind';
%%% g is mode of 
%% Read change data from Excell
excelFile = 'data/data512.xlsx'; 
LookupT1 = xlsread(excelFile);
C1=[10;1;7;12;14];
state1=42;
excelFile = 'data/data513.xlsx'; 
LookupT2 = xlsread(excelFile);
C2=[22,7,21,39,48];
state2=105;
%%
ans_edge=testforhistogram(img);
edge=ans_edge(:);
szl=size(edge,1);
%count_ones = sum(edge(:) == 1);
%count_zeros=szl-count_ones;

%%% s secret message
%s=randi([0,state-1],1,ls);
Ext1=[];
sec=[];
count_sec=1;
count_ext=1;
flag=[];

for i=1:n:l-rem_ls
   edg_det(:)=edge(i:i+(n-1));
   if sum(edg_det(:) == 1)>2
       tt= double(img_vec(i:i+(n-1)));
       Ext1(count_ext)=Extraction(tt,C2,n,state2);
       %Ext1(count)=Extraction(img_vec(i:i+(n-1)),C,n,state);
       count_ext=count_ext+1;    
       sec(count_sec)=randi([0,state2-1]);
       flag(count_sec)=1;
       count_sec=count_sec+1;
   else
       tt= double(img_vec(i:i+(n-1)));
       Ext1(count_ext)=Extraction(tt,C1,n,state1);
       %Ext1(count)=Extraction(img_vec(i:i+(n-1)),C,n,state);
       count_ext=count_ext+1;
       sec(count_sec)=randi([0,state1-1]);
       flag(count_sec)=0;
       count_sec=count_sec+1;
   end
end


%% Embedding
%
chg=[];
img_stego=[];
det=[];
count1=1;
for jj=1:1:ls
    if flag(jj)==1
        det(jj)=mod((sec(jj)-Ext1(jj)),state2);
        % chg=lookupT232(det(jj));
        for ic=1:1:state2
            if( det(jj)== LookupT2(ic,6))
                chg(1)=LookupT2(ic,1);
                chg(2)=LookupT2(ic,2);
                chg(3)=LookupT2(ic,3);
                chg(4)=LookupT2(ic,4);
                chg(5)=LookupT2(ic,5);
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
        img_stego(count1)=img_vec(count1)+chg(5);
        count1=count1+1;

    else
        det(jj)=mod((sec(jj)-Ext1(jj)),state1);
        % chg=lookupT232(det(jj));
        for ic=1:1:state1
            if( det(jj)== LookupT1(ic,6))
                chg(1)=LookupT1(ic,1);
                chg(2)=LookupT1(ic,2);
                chg(3)=LookupT1(ic,3);
                chg(4)=LookupT1(ic,4);
                chg(5)=LookupT1(ic,5);
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
        img_stego(count1)=img_vec(count1)+chg(5);
        count1=count1+1;

    end
                                                                                                            
end

img_stego1=[img_stego,remind];
img_stego1=uint8(img_stego1(:));
%disp(img_stego(:)-img_vec(:))

Stego = reshape(img_stego1, L(1), L(2));
%}
%% Histogram
%
figure;
%Histogram(img_vec,img_stego1);
hist1 = imhist(img);
hist2 = imhist(Stego);

plot(hist1,LineWidth=1.5);
hold on;
plot(hist2,LineWidth=1.5);xlabel('Value',FontSize=11,FontWeight='bold');ylabel('Histogrm of two Image',FontSize=11,FontWeight='bold');
legend('Orginal Image','Stego Image')

%
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
%
Ext2=[];
count2=1;
for i=1:n:l-rem_ls
    if flag(count2)==1
        tt1= double(img_stego1(i:i+(n-1)));
        Ext2(count2)=Extraction(tt1,C2,n,state2);
        count2=count2+1;
    else
        tt1= double(img_stego1(i:i+(n-1)));
        Ext2(count2)=Extraction(tt1,C1,n,state1);
        count2=count2+1;

    end
    
end

sum_Ext=sum(Ext2(:)-sec(:));
for i=1:1:32768
%if(Ext2(1,i)~=s(1,i))
%disp(i);
%disp(Ext2(i));
%disp(s(i));
%end
end


%}