function [prediction, accuracy] = eval_fisher_discriminant_analysis(model, feature, label)
%eval_fisher_discriminant_analysis
%returns the predictions for the feautures with the model and the accuracy
%   model: model created by train_fisher_discriminant_analysis
%   feature: feature vector for the evaluation with size(feature) = [nbMatches, nbFeatures]
%   label: ground truth label for the features

[prediction, ~] = predict_fisher_discriminant_analysis(model, feature);
accuracy = length(find(label == prediction)) / length(label);
disp("accuracy of Fischer discriminant analysis is " + accuracy)

end

