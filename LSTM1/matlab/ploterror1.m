clear; clc; cla(gca);

data = csvread("trainingError.txt");
%data = csvread("validationError.txt");
%data = csvread("debug1.txt");
%data = csvread("param.txt");

error = data(:, 1);

x_ = linspace(1, size(error, 1), size(error, 1));
x = transpose(x_);

plot(x, error, 'Color', [.945 .639 .416]); hold on;
%plot(data(:,1), data(:,2));

ma = movmean(error, 100);
plot(x, ma, 'Color', [.729 .271 .302], 'Linewidth', 2);
