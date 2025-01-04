function [data] = prepare_data_stepwise_regression(data)
%prepare_data_stepwise_regression
%returns the resulting data
%   data: data with size(data) = [nbSamples, nbChannels, nbMatches]

if size(size(data), 2) ~= 3
    error("data does not have 3 dimensions")
end
[nbSamples, nbChannels, nbMatches] = size(data);

data = reshape(data, [nbSamples*nbChannels, nbMatches])';

end

