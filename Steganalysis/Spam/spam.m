clc;
clear all;
close all;

%% Extract feature cover & stego img for train
Feature_cover=[];
for i=1:8
img=strcat('cover_pic\a',num2str(i),'.jpg');
Feature_cover(i,:)=spam686(img);
end
 
Feature_stego=[];
for j=1:8
img=strcat('stego_pic\w',num2str(j),'.jpg');
Feature_stego(j,:)=spam686(img);
end
 
Feature=[];
Label=[];
 
Feature(1:2,:)=Feature_cover(:,:);
Feature(3:4,:)=Feature_stego(:,:);
Label_cover=ones(2,1);
Label_stego=-ones(2,1);
Label(1:2,:)=Label_cover;
Label(3:4,:)=Label_stego;
train=Feature;
%
%% Extract feature for test
Feature_test=[];
for k=1:2
img=strcat('testi_pic\t',num2str(k),'.jpg');
Feature_test(k,:)=spam686(img);
end
test=Feature_test;
%% Training & Ttest
 
SVMModel= fitcsvm(Feature, Label);
%CompactSVMModel = SVMModel.Trained{1};
[label,score] = predict(SVMModel,test);


for i=1:1:len(label)
     if label(i,:)< 0
         cont=cont+1;
 end
end
 rate=(cont/200)*100



