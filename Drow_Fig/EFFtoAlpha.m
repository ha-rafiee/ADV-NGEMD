clc;
clear all;
close all;

x=[ 1.445324132,1.262,0.9272,0.8152,0.7447,0.7340,0.6984, 0.6821, 0.6309,0.4515,0.37799];
y=[3.80,3.566,3.1455,2.9441,2.7755,2.6725,2.5727,2.5658,2.37,2.0442,1.8071];
%xi = [x(1):0.0000001:x(end)];
%vid=interp1(x,y,xi,'pchip');
%figure;
%
x_emd=[0.86,1.068,1.262];
y_emd=[2.9,3.275,3.566];


x_femd=[1];
y_femd=[1];

x_de=[0.54];
y_de=[1.85];

x_iemd=[0.66];
y_iemd=[1.5];

x_gemd=[0.66666];
y_gemd=[1.5];

x_egemd=[0.666];
y_egemd=[1.5];

x_multi=[0.6666];
y_multi=[1];

x_value=[0.5];
y_value=[1];

x_hemd=[0.632];
y_hemd=[1.58];

x_emd2=[0.6329];
y_emd2=[1.58];

x_msd=[0.869];
y_msd=[1.15];


x_appm=[0.5];
y_appm=[1.33];

x_mpemd=[1];
y_mpemd=[1];

x_aemd=[0.5];
y_aemd=[1.0];

x_2F=[0.4];
y_2F=[1];

x_catal=[1];
y_catal=[0.125];

x_kirs=[0.75];
y_kirs=[0.66];

x_rgemd=[0.5];
y_rgemd=[0.85];

x_pvd=[0.4];
y_pvd=[0.45];

x_DualImage=[1];
y_DualImage=[1];
figure;

plot(x,y,'r', LineWidth=3)
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



plot(x_appm,y_appm,'|','Color','#708090','MarkerSize',10,LineWidth=2);
plot(x_mpemd,y_mpemd,'D','Color','#556B2F','MarkerSize',10,LineWidth=2);
plot(x_aemd,y_aemd,'_','Color','#A0522D','MarkerSize',10,LineWidth=2);

plot(x_2F,y_2F,'*','Color','#006400','MarkerSize',10,LineWidth=2);
plot(x_catal,y_catal,'.','Color','#8B0000','MarkerSize',10,LineWidth=2);
plot(x_kirs,y_kirs,'P','Color','#7FFF00','MarkerSize',10,LineWidth=2);
plot(x_rgemd,y_rgemd,'p','Color','#00008B','MarkerSize',10,LineWidth=2);
plot(x_pvd,y_pvd,'s','Color','#00BFFF','MarkerSize',10,LineWidth=2);
plot(x_DualImage,y_DualImage,'hexagram','Color','#8B4513','MarkerSize',10,LineWidth=2);
xlabel('1/Payload','FontSize',13,FontWeight='bold');
ylabel('Efficiency','FontSize',13,FontWeight='bold');

% Create a legend
legend({'ProposedMethod', 'EMD', 'FEMD', 'DE', 'IEMD', 'GEMD', 'Enhanced GEMD', 'Multi-bit Encoding', 'Pixel Value Adjustment', 'Hypercube EMD', 'EMD-2', 'MSD BASED',  'APPM', 'MPEMD', 'AEMD', '2-Functions', 'Catalan Base', 'Kirsch Base', 'RGEMD', 'Pixel Value Differencing','Dual-Image'},'FontSize',12,'FontWeight','bold');

% Save the figure
saveas(gcf, 'myfig.png', 'png');

% Display the figure
hold off;
%}
