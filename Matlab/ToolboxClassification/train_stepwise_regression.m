function [coefficients, b0, finalmodel, pval] = train_stepwise_regression(data, label)
%train_stepwise_regression
%returns the parameters of the created classification model
%   data: data for the training with size(data) = [nbSamples*nbChannels, nbMatches]
%   label: ground truth label for the data

[W, ~, pval, finalmodel, stats] = stepwisefit(data, label, 'maxiter', 60, 'display', 'off', 'penter', 0.1, 'premove', 0.15);

b0 = stats.intercept; % bias
coefficients = W(finalmodel~=0); % coefficients

end

