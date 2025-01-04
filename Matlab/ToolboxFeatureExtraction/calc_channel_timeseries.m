function [time_vector, channel_timeseries] = calc_channel_timeseries(data, sampling_frequency, window_size, step_size)
%calc_channel_timeseries
%returns the time_vector and channel_timeseries with size(channel_timeseries) = [nbSteps, nbChannels, nbTrials]
%   data: recorded signals with size(data) = [nbSamples, nbChannels, nbTrials]
%   sampling_frequency: frequency at which the signals were sampled
%   window_size: size of the window, needs to be a power of two value
%   step_size: size of the steps, needs to be a power of two value

log_window_size = log(window_size)/log(2);
log_step_size = log(step_size)/log(2);
if round(log_window_size) - log_window_size ~= 0 || round(log_step_size) - log_step_size ~= 0
    error("window_size and step_size need to be power of two values")
end

if size(size(data), 2) ~= 3
    error("data does not have 3 dimensions")
end    
[nbSamples, nbChannels, nbTrials] = size(data);

nbSteps = (nbSamples - window_size + step_size) / step_size;
channel_timeseries = zeros(nbSteps, nbChannels, nbTrials);
for i = 1:nbSteps
    factor = (i-1)*step_size;
    
    for t = 1:nbTrials
        window = data(1+factor:window_size+factor, :, t);
        channel_timeseries(i, :, t) = mean(window.^2);
    end
end

time_vector = window_size/2:step_size:nbSamples-window_size/2;
time_vector = time_vector ./ sampling_frequency;

end

