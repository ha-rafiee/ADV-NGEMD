function Histogram(vector1,vector2)
% Create a histogram for vector1
histogram(vector1, 'Normalization', 'pdf', 'EdgeColor', 'b', 'FaceColor', 'b', 'DisplayName', 'Vector 1');
hold on;

% Create a histogram for vector2
histogram(vector2, 'Normalization', 'pdf', 'EdgeColor', 'r', 'FaceColor', 'r', 'DisplayName', 'Vector 2');

% Add labels and legend
xlabel('Value');
ylabel('Probability Density');
title('Histogram of Two Vectors');
legend;

% Optionally, adjust axis limits for better visibility
% xlim([-3, 3]);

% Hold off to end overlay mode
hold off;
end