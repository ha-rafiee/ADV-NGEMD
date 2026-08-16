clc;
close all;
clear all;


x = [0.14741, 0.180288, 0.195, 0.245342, 0.289377, 0.315789, 0.357143, 0.431034, 1.54, 1.78571, 2, 2.32353, 2.58621, 3, 3.89474, 4.92308];
y = [0.211718, 0.408906, 0.507768, 0.832, 1.09279, 1.36838, 1.75008, 2.59384, 1.80893, 1.72126, 1.65363, 1.56712, 1.50296, 1.41944, 1.27968, 1.15363];

%
xi = [x(1):0.0000001:x(end)];
vid=interp1(x,y,xi,'pchip');
%plot(x,y,'b*')
x_emd=[0.630929753571457,0.861353116146786,1.06862156132407,1.26185950714291];
y_emd=[1.94117469410622,1.29799726475868,1.01076329434746,0.840553299139738];

x_femd=[1];
y_femd=[1.63290086676535];

x_de=[0.862068965517241];
y_de=[1.83196768555895];

x_iemd=[0.6666666666666];
y_iemd=[1.89977023333431];

x_gemd=[0.75187969924812];
y_gemd=[1.80431126474017];

x_egemd=[0.598802395209581];
y_egemd=[1.58736313757108];

x_multi=[0.75];
y_multi=[1.75145327664224];

x_value=[0.5];
y_value=[0.5914];

x_hemd=[0.628930817610063];
y_hemd=[1.94920242891259];

x_emd2=[0.630929753571457];
y_emd2=[1.941175];

x_msd=[0.867194478953663];
y_msd=[1.77092662529155];

x_eemdhw=[0.333];
y_eemdhw=[1.11066];



x_appm=[0.655];
y_appm=[1.9];

x_mpemd=[1];
y_mpemd=[1.15470053837925];

x_aemd=[0.959877135726627];
y_aemd=[1.62646294228484];

x_2F=[0.4];
y_2F=[1.5045264537248];

x_catal=[1];
y_catal=[1.05793703233134];

x_kirs=[0.540715907862009];
y_kirs=[1.27925672947392];

x_rgemd=[0.5];
y_rgemd=[1.35048515685177];

x_pvd=[0.6535];
y_pvd=[0.7964];

figure;

plot(xi,vid,'r', LineWidth=2)
hold on

plot(x_emd, y_emd, '--gs', 'Color', 'green','MarkerSize',10,LineWidth=2);
plot(x_femd, y_femd, '.', 'Color', 'black','MarkerSize',10,LineWidth=2);
plot(x_de, y_de, 'o', 'Color', 'yellow','MarkerSize',10,LineWidth=2);
plot(x_iemd,y_iemd,'v','Color','#FF7F50','MarkerSize',10,LineWidth=2);
plot(x_gemd,y_gemd,'^','Color','#9370DB','MarkerSize',10,LineWidth=2);
plot(x_egemd,y_egemd,'>','Color','black','MarkerSize',10,LineWidth=2);
plot(x_multi,y_multi,'<','Color','#800080','MarkerSize',10,LineWidth=2);


plot(x_value,y_value,'diamond','Color','#0072BD','MarkerSize',10,LineWidth=2);
plot(x_hemd,y_hemd,'s','Color','#EE82EE','MarkerSize',10,LineWidth=2);
plot(x_emd2,y_emd2,'h','Color','#008080','MarkerSize',10,LineWidth=2);
plot(x_msd,y_msd,'x','Color','#FA8072','MarkerSize',10,LineWidth=2);


plot(x_eemdhw,y_eemdhw,'hexagram','Color','#8B4513','MarkerSize',10,LineWidth=2);
plot(x_appm,y_appm,'|','Color','#708090','MarkerSize',10,LineWidth=2);
plot(x_mpemd,y_mpemd,'D','Color','#556B2F','MarkerSize',10,LineWidth=2);
plot(x_aemd,y_aemd,'_','Color','#A0522D','MarkerSize',10,LineWidth=2);

plot(x_2F,y_2F,'*','Color','#006400','MarkerSize',10,LineWidth=2);
plot(x_catal,y_catal,'.','Color','#8B0000','MarkerSize',10,LineWidth=2);
plot(x_kirs,y_kirs,'P','Color','#7FFF00','MarkerSize',10,LineWidth=2);
plot(x_rgemd,y_rgemd,'p','Color','#00008B','MarkerSize',10,LineWidth=2);
plot(x_pvd,y_pvd,'s','Color','#00BFFF','MarkerSize',10,LineWidth=2);
xlabel('Inverse Alpha');
ylabel('Efficiency');

% Create a legend
legend({'Bound', 'EMD', 'FEMD', 'DE', 'IEMD', 'GEMD', 'Enhanced GEMD', 'Multi-bit Encoding', 'Pixel Value Adjustment', 'Hypercube EMD', 'EMD-2', 'MSD BASED', 'EEMDHW', 'APPM', 'MPEMD', 'AEMD', '2-Functions', 'Catalan Base', 'Kirsch Base', 'RGEMD', 'Pixel Value Differencing'},'FontSize',12,'FontWeight','bold');

% Save the figure
saveas(gcf, 'myfig.png', 'png');

% Display the figure
hold off;

%%
%{

[xx,is] = sort(x(:));
yy = y(is);
yy = yy(:);
dx = diff(xx);
dy = diff(yy);
y0 = yy(1:end-1);
n = numel(xx)-1;
coefs = [-2*dy./(dx.^3), 3*dy./(dx.^2), 0*dy, y0];
pp = struct('form', 'pp',...
    'breaks', xx(:)',...
    'coefs', coefs,...
    'pieces', n, ...
    'order', 4,...
    'dim', 1);
figure
xi = linspace(min(x),max(x));
yi = ppval(pp,xi);
%plot(x,y,'b-o',xi,yi,'r');
plot(xi,yi,'r');
xlim([min(x),max(x)])
grid on

%}
%%
%{
a=[0 1; 1 1];%a=[1 0; 1 1] is ok
b=inv(a);
c=[];
f=[3;4];
c=b*f;
%}
%{
a=[1 0 0; 1 0 1;1 1 0];
b=inv(a);
c=[];
f=[1;10;4];
c=mod((b*f),27);
%}
