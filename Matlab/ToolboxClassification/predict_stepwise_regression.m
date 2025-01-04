function [prediction, distance] = predict_stepwise_regression(coefficients, b0, finalmodel, data)
%predict_stepwise_regression
%returns the predictions and the distances for the data with the model
%   coefficients: part of model created by train_stepwise_regression
%   b0: part of model created by train_stepwise_regression
%   finalmodel: part of model created by train_stepwise_regression
%   data: data for the prediction with size(data) = [nbSamples*nbChannels, nbMatches]

data = data(:,finalmodel~=0); % only use data included by the model
distance = (sum(data' .* coefficients) + b0)';
prediction = distance;
prediction(prediction>=0) = 1;
prediction(prediction<0) = -1;

end

