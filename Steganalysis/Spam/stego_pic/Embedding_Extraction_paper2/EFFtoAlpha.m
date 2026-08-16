clc;
clear all;
close all;

x=[0.630954634,0.430663221,0.356214156];
y=[1.6376,1.9349,2.3774];
%xi = [x(1):0.0000001:x(end)];
%vid=interp1(x,y,xi,'pchip');
%figure;
%
x_emd=[0.86];
y_emd=[2.9];


x_femd=[1];
y_femd=[1];


x_hemd=[0.632];
y_hemd=[1.58];


x_appm=[0.5];
y_appm=[1.33];


x_2F=[0.4];
y_2F=[1];

figure;

%plot(x,y,'r', LineWidth=3)
plot(x, y, '--gs', 'Color', 'Red','MarkerSize',10,LineWidth=2);

hold on

plot(x_emd, y_emd, 'gs', 'Color', 'Blue','MarkerSize',10,LineWidth=2);
plot(x_femd, y_femd, 'gs', 'Color', '#9370DB','MarkerSize',10,LineWidth=2);


plot(x_hemd,y_hemd,'gs','Color','#EE82EE','MarkerSize',10,LineWidth=2);
plot(x_appm,y_appm,'gs','Color','#00BFFF','MarkerSize',10,LineWidth=2);
plot(x_2F,y_2F,'gs','Color','green','MarkerSize',10,LineWidth=2);

xlabel('Invers Payload','FontSize',13,FontWeight='bold');
ylabel('Efficiency','FontSize',13,FontWeight='bold');

% Create a legend
legend({'ProposedMethod', 'EMD', 'FEMD',  'HEMD',  'APPM', '2-Functions', },'FontSize',12,'FontWeight','bold');

% Save the figure
saveas(gcf, 'myfig.png', 'png');

% Display the figure
hold off;
%}
