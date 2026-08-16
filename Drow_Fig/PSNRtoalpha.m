clc;
clear all;
close all;

x=[0.4,1.07846,1.2267,1.3428,1.3624,1.4319,1.4661,1.584963,1.9782,2.2146];
y=[58.348,52.778,51.93,51.2822,51.0602,50.6755,50.5569,49.889,47.7910,47.4218];

x_emd=[0.4,1.16];
y_emd=[58.348,52.101];

x_emd2=[1.18];
y_emd2=[52];

x_aemd=[1.0547,2.027];
y_aemd=[51.82,46.28];

x_femd=[1.5,2];
y_femd=[49.88,46.74];

x_gemd=[1.16,1.1];
y_gemd=[50.72,51.19];

x_hemd=[1.5,2];
y_hemd=[49.88,46.75];

x_appm=[1.16,2.679];
y_appm=[52.101,42.89];

x_do=[0.5,1];
y_do=[47.2,45.3];

figure;

plot(x,y,'r', LineWidth=3)
hold on

plot(x_emd, y_emd, '--gs', 'Color', 'green','MarkerSize',10,LineWidth=2);
plot(x_emd2,y_emd2,'P','Color','#008080','MarkerSize',10,LineWidth=3);
plot(x_aemd,y_aemd,'--h','Color','#A0522D','MarkerSize',10,LineWidth=2);
plot(x_femd, y_femd, '--x', 'Color', 'black','MarkerSize',10,LineWidth=3);
plot(x_gemd,y_gemd,'--^','Color','#9370DB','MarkerSize',10,LineWidth=2);
plot(x_hemd,y_hemd,'--s','Color','#EE82EE','MarkerSize',10,LineWidth=2);
plot(x_appm,y_appm,'--|','Color','Blue','MarkerSize',10,LineWidth=2);
plot(x_do,y_do,'--*','Color','#006400','MarkerSize',10,LineWidth=2);
xlabel('Payload','FontSize',13,FontWeight='bold');
ylabel('PSNR','FontSize',13,FontWeight='bold');

legend({'ProposedMethod', 'EMD','EMD-2','AEMD', 'FEMD', 'GEMD','HEMD', 'APPM','Dual-Image'},'FontSize',12,'FontWeight','bold');

% Save the figure
saveas(gcf, 'myfig.png', 'png');

% Display the figure
hold off;
%}