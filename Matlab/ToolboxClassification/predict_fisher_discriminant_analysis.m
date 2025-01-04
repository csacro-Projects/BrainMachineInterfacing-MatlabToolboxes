function [prediction, score] = predict_fisher_discriminant_analysis(model, feature)
%predict_fisher_discriminant_analysis
%returns the predictions and the scores for the feautures with the model
%   model: model created by train_fisher_discriminant_analysis
%   feature: feature vector for the prediction with size(feature) = [nbMatches, nbFeatures]

[prediction, score] = predict(model, feature);

end

