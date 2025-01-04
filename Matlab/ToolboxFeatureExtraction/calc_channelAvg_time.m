function [time_vector, channel_average_time] = calc_channelAvg_time(data, sampling_frequency)
%calc_channelAvg_time
%returns the time_vector and the channel_average_time with size(channel_average_time) = [nbSamples, nbChannels]
%   data: recorded signals with size(data) = [nbSamples, nbChannels, nbTrials]
%   sampling_frequency: frequency at which the signals were sampled

if size(size(data), 2) ~= 3
    error("data does not have 3 dimensions")
end
[nbSamples, ~, ~] = size(data);

dt = 1/sampling_frequency;
time_vector = 0:dt:nbSamples*dt-dt;
channel_average_time = squeeze(mean(data, 3));

end

