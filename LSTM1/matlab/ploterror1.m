clear; clc;

data = csvread("trainingError.txt");
%data = csvread("validationError.txt");
%data = csvread("debug1.txt");
%data = csvread("param.txt");

error = data(:, 1);

x = linspace(1, size(error, 1), size(error, 1));

plot(x, error);
%plot(data(:,1), data(:,2));
