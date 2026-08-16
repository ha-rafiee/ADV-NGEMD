clc;
close all;
clear all;
a=audioread("A.wav");
s=size(a);
b=rand(6,1);

a1=a;
a1(44300:44305,1)=a1(44300:44305,1)+b;


snr(abs(a)-abs(a1),a)
SNR=10*log10(sum(a(:).^2)/(sum((a(:)-a1(:)).^2)))