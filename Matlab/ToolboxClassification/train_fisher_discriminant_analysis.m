function [model] = train_fisher_discriminant_analysis(feature, label)
%train_fisher_discriminant_analysis
%returns the created classification model
%   feature: feature vector for the training with size(feature) = [nbMatches, nbFeatures]
%   label: ground truth label for the features
%   eval_train: whether model should be evaluated on training data

model = fitcdiscr(feature, label);

end

