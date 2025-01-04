function [] = plot_stepwise_regression(coefficients, b0, finalmodel, pval, data, label)
%plot_stepwise_regression
%plots the data and the decision boundary
%   coefficients: part of model created by train_stepwise_regression
%   b0: part of model created by train_stepwise_regression
%   finalmodel: part of model created by train_stepwise_regression
%   pval: p-value for the significance
%   data: data for the evaluation with size(data) = [nbSamples*nbChannels, nbMatches]
%   label: ground truth label for the data

data = data(:,finalmodel~=0); % only use data included by the model
figure('Name', 'Classification with stepwise Regression (two most discriminative features)')
hold on;
% feature points
[~, significance_pos] = sort(pval(finalmodel~=0), 'ascend'); % the smaller p-value the more sigificant
plot(data(label==-1,significance_pos(1)), data(label==-1,significance_pos(2)), 'o', 'color', [1 1 1], 'MarkerFaceColor', 'r', 'markerSize', 8)
plot(data(label==1,significance_pos(1)), data(label==1,significance_pos(2)), 'o', 'color', [1 1 1], 'MarkerFaceColor', 'b', 'markerSize', 8)
% decision boundary
x = linspace(min(data(:,significance_pos(1))), max(data(:,significance_pos(1))), 10);
y = -x * (coefficients(significance_pos(1)) / coefficients(significance_pos(2))) - (b0 / coefficients(significance_pos(2))); % x*wx + y*wy + b0 = 0
plot(x, y, 'k', 'linewidth', 2)
hold off;

end

