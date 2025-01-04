function [r2] = calc_correlation(data1, data2)
%calc_correlation
%returns the frequency_vector and the correlation r2 between data and compareData for all frequencies in the vector
%   data1: recorded signals with size(data) = [nbSamples, nbChannels, nbTrials]
%   data2: recorded signals with size(compareData) = [nbSamples, nbChannels, nbCompareTrials]

if size(size(data1), 2) ~= 3
    error("data1 does not have 3 dimensions")
end
[nbSamples, nbChannels, ~] = size(data1);
if size(size(data2), 2) ~= 3
    error("data2 does not have 3 dimensions")
end
[nbCompareSamples, nbCompareChannels, ~] = size(data1);
if nbSamples ~= nbCompareSamples || nbChannels ~= nbCompareChannels
    error("data1 and data2 do not have the same amount of samples or channels")
end

labels = cat(1, zeros(size(data1, 3), 1), ones(size(data2, 3), 1));
data = cat(3, data1, data2);

r2 = [];
for channel=1:nbChannels
    r2 = cat(1, r2, corr(labels, squeeze(data(:, channel, :))').^2);
end

end

