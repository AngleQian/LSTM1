clear;
clc;

%data = csvread('trainingOutput.txt');
data = csvread('validationOutput.txt');

trueprice = data(:, 1);
outputprice = data(:, 2);

x = linspace(1, size(trueprice, 1), size(trueprice, 1));

plot(x, trueprice, 'Color', 'b'); hold on;
plot(x, outputprice, 'Color', 'r');

set(legend, 'interpreter', 'latex', 'FontSize', 18);
legend('True', 'Predicted', 'Location', 'southeast');

set(gca, 'LineWidth', .5, 'FontSize', 18, 'TickLabelInterpreter', 'latex');
set(gca, 'XGrid', 'on', 'XMinorGrid', 'off', 'YGrid', 'on', 'YMinorGrid', 'off');

xlabel('Days since 2006/01/03', 'interpreter', 'latex', 'FontSize', 18);
ylabel('S\&P 500', 'interpreter', 'latex', 'FontSize', 18);
title('Predicting $t+1$ Daily Close Price of the S\&P 500', 'interpreter', 'latex', 'FontSize', 22);
