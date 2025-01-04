function [feature, label] = prepare_fisher_discriminant_analysis(data_class1, data_class2, feature_function)
%prepare_fisher_discriminant_analysis
%returns the resulting features and labels
%   data_class1: data for the first class with size(data_class1) matches input of feature_function
%   data_class2: data for the second class with size(data_class2) matches input of feature_function
%   feature_function: function to transform data into features with size [nbMatches, nbFeatures]

feature_class1 = feature_function(data_class1);
feature_class2 = feature_function(data_class2);

if size(size(feature_class1), 2) ~= 2
    error("feature_class1 does not have 2 dimensions")
end
if size(size(feature_class2), 2) ~= 2
    error("feature_class2 does not have 2 dimensions")
end
[nbMatchesClass1, nbFeatures1] = size(feature_class1);
[nbMatchesClass2, nbFeatures2] = size(feature_class2);
if nbFeatures1 ~= nbFeatures2
    error("amount of features must be the same for both classes")
end

feature = [feature_class1; feature_class2];
label = [-ones(nbMatchesClass1, 1); ones(nbMatchesClass2, 1)];

end

