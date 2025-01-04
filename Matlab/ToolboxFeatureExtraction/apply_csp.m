function [feature] = apply_csp(data, cspMatrix, nbFilters)
%apply_csp
%returns the features calculated with csp algorithm and the labels
%   data: data to apply csp to with size(data) = [nbSamples, nbChannels, nbMatchesClass]
%   cspMatrix: matrix calculated for the csp algorithm (e.g. via calc_csp.m)
%   nbFilters: specify the number of features which is 2*nbFilter (e.g. nbFilters = 1)

if size(size(data), 2) ~= 3
    error("data does not have 3 dimensions")
end
[nbSamples, ~, nbMatches] = size(data);
nbFeatures = 2*nbFilters;

% project EEG signals (only those that are relevant for feature extraction)
cspMatrix = cspMatrix([1:nbFilters end-nbFilters+1:end], :);
cspMatrix_transposed = cspMatrix';
data_projected = zeros(nbSamples, nbFeatures, nbMatches);
for m = 1:nbMatches
    data_projected(:, :, m) = data(:, :, m) * cspMatrix_transposed;
end

% extract features based on variance
feature = zeros(nbMatches, nbFeatures);
for m = 1:nbMatches
    for f = 1:nbFeatures
        variance = var(data_projected(:, f, m));
        feature(m, f) = log(variance);
    end  
end

end