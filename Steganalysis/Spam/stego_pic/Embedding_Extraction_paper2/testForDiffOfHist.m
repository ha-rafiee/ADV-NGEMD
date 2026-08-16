clc;
clear all;
close all;
img=imread("img/lena.jpg");
L=size(img);
l=L(1)*L(2);
img_vec=img(:);
%%% n lenght of the group/ z chane of each pixel / k the number of pixel
%%% can be change/ c array is coffisient
n=2;
z=1;
k=1;
C=[4;3];
ls=floor(l/n);
rem_ls=mod(l,n);

%%% g is mode of 
state=5;

%%% s secret message
%s=randi([0,state-1],1,ls);
excelFile = 'sec.xlsx'; 
s = xlsread(excelFile);

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
 chg=lookupT211_43(det(jj));
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
figure;
hist1 = imhist(img);
hist2 = imhist(Stego);

plot(hist1,LineWidth=1.5);
hold on;
plot(hist2,LineWidth=1.5);xlabel('Value',FontSize=11,FontWeight='bold');ylabel('Histogrm of two Image',FontSize=11,FontWeight='bold');
legend('Orginal Image','Stego Image')
absdiff = abs(hist2- hist1);
sumOfdiff=sum(absdiff(:));

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


%}


%{
x1 = randn(1, 1e3);
x2 = randn(1, 1e3) + 0.25;
E = -5:0.25:5;
subplot(3, 1, 1)
h1 = histogram(x1, E);

subplot(3, 1, 2)
h2 = histogram(x2, E);

% Set the limits of the two axes containing each histogram to be the same

% First get the axes handles
ax1 = ancestor(h1, 'axes');
ax2 = ancestor(h2, 'axes');

% Set the X limits for both axes to be [-5 5] since that's the limit of our edges
xlim(ax1, [-5 5]);
xlim(ax2, [-5 5]);

% Get the Y limits of each axes
y1 = ylim(ax1);
y2 = ylim(ax2);

% The lower Y limit of each axes should be 0.
% The upper Y limit should be the greater of the two that histogram chose
%   based on the largest BinCount
consistentYlimits = [0, max(y1(2), y2(2))];
ylim(ax1, consistentYlimits);
ylim(ax2, consistentYlimits);

%absdiff = abs(h1.BinCounts - h2.BinCounts);
subplot(3, 1, 3);
h3 = histogram('BinCounts', absdiff, 'BinEdges', E);

% Set the limits of the third axes to be the same as those of the first two axes
ax3 = ancestor(h3, 'axes');
xlim(ax3, [-5 5])
yl  im(ax3, consistentYlimits);
%}