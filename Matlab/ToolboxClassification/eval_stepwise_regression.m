function [prediction, accuracy] = eval_stepwise_regression(coefficients, b0, finalmodel, pval, data, label)
%predict_stepwise_regression
%returns the predictions for the data with the model and the accuracy
%   coefficients: part of model created by train_stepwise_regression
%   b0: part of model created by train_stepwise_regression
%   finalmodel: part of model created by train_stepwise_regression
%   pval: p-value for the significance
%   data: data for the evaluation with size(data) = [nbSamples*nbChannels, nbMatches]
%   label: ground truth label for the data

[prediction, ~] = predict_stepwise_regression(coefficients, b0, finalmodel, data);
accuracy = length(find(label == prediction)) / length(label);
disp("accuracy of stepwise Regression is " + accuracy)

end

