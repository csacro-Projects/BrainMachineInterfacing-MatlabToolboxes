function [feature, label] = prepare_fisher_discriminant_analysis_csp(data_class1, data_class2, cspMatrix, nbFilters)
%prepare_fisher_discriminant_analysis
%returns the resulting features and labels
%   data_class1: data for the first class with size(data) = [nbSamples, nbChannels, nbMatchesClass1]
%   data_class2: data for the first class with size(data) = [nbSamples, nbChannels, nbMatchesClass2]
%   cspMatrix: matrix calculated for the csp algorithm (e.g. via calc_csp.m)
%   nbFilters: specify the number of features which is 2*nbFilter (e.g. nbFilters = 1)

if size(size(data_class1), 2) ~= 3
    error("data_class1 does not have 3 dimensions")
end
if size(size(data_class2), 2) ~= 3
    error("data_class2 does not have 3 dimensions")
end
[~, ~, nbMatchesClass1] = size(data_class1);
[~, ~, nbMatchesClass2] = size(data_class2);

label = [-ones(nbMatchesClass1, 1); ones(nbMatchesClass2, 1)];

data = cat(3, data_class1, data_class2);
feature = apply_csp(data, cspMatrix, nbFilters);

end

