clear;
clc;


data = csvread('trainingOutput.txt');

trueprice = data(:, 1);
outputprice = data(:, 2);

x = linspace(0, 100, size(data, 1));

plot(x, trueprice, 'Color', 'b'); hold on;
plot(x, outputprice, 'Color', 'r');

title('Predicting $t+1$ Daily Close Price of the S\&P 500', 'interpreter', 'latex', 'FontSize', 15);