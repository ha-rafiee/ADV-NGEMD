clc;
clear all;
close all;

x=[1.5849,2.322,2.8073];
y=[49.9,45.158,42.14];

x_emd=[1.16];
y_emd=[52.101];



x_femd=[2];
y_femd=[46.74];


x_hemd=[1.5,2];
y_hemd=[49.88,46.75];

x_appm=[2.679];
y_appm=[42.89];


x_2func=[2.5];
y_2func=[43.73];

figure;

plot(x,y,'r', LineWidth=3)
hold on

plot(x_emd, y_emd, '--gs', 'Color', 'green','MarkerSize',10,LineWidth=2);
plot(x_femd, y_femd, '--x', 'Color', 'black','MarkerSize',10,LineWidth=3);
plot(x_hemd,y_hemd,'--s','Color','#EE82EE','MarkerSize',10,LineWidth=2);
plot(x_appm,y_appm,'--|','Color','Blue','MarkerSize',10,LineWidth=2);
plot(x_2func,y_2func,'--*','Color','#006400','MarkerSize',10,LineWidth=2);
xlabel('Payload','FontSize',13,FontWeight='bold');
ylabel('PSNR','FontSize',13,FontWeight='bold');

legend({'ProposedMethod', 'EMD', 'FEMD','HEMD', 'APPM','Two Function'},'FontSize',12,'FontWeight','bold');

% Save the figure
saveas(gcf, 'myfig.png', 'png');

% Display the figure
hold off;
%}