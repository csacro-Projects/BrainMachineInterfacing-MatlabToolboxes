function [data, label] = prepare_stepwise_regression(data_class1, data_class2)
%prepare_stepwise_regression
%returns the resulting data and labels
%   data_class1: data of the first class with size(data_class1) = [nbSamples, nbChannels, nbMatchesClass1]
%   data_class2: data of the second class with size(data_class2) = [nbSamples, nbChannels, nbMatchesClass2]

if size(size(data_class1), 2) ~= 3
    error("data_class1 does not have 3 dimensions")
end
if size(size(data_class2), 2) ~= 3
    error("data_class2 does not have 3 dimensions")
end
[nbSamples, nbChannels, nbMatchesClass1] = size(data_class1);
[nbSamples2, nbChannels2, nbMatchesClass2] = size(data_class2);
if nbSamples ~= nbSamples2 || nbChannels ~= nbChannels2
    error("amount of samples and channels must be the same for both classes")
end

data = [
    prepare_data_stepwise_regression(data_class1);...
    prepare_data_stepwise_regression(data_class2)
];
label = [-ones(nbMatchesClass1, 1); ones(nbMatchesClass2, 1)];

end

