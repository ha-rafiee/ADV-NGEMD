function Ext=Extraction(v_img,C,n,g)
%%% v_img is vector of img with n length
%%% C is coffisient of embedding
%%% n the length of group
%%% g is Mode

temp=0;
for i=1:n
temp=temp+v_img(i)*C(i);
end
Ext=mod(temp,g);
end