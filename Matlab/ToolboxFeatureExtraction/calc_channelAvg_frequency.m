function [frequency_vector, channel_average_frequency] = calc_channelAvg_frequency(data, sampling_frequency)
%calc_channelAvg_frequency
%returns the frequency_vector and channel_average_frequency with size(channel_average_frequency) = [nbSamples/2, nbChannels]
%   data: recorded signals with size(data) = [nbSamples, nbChannels, nbTrials]
%   sampling_frequency: frequency at which the signals were sampled

if size(size(data), 2) ~= 3
    error("data does not have 3 dimensions")
end
[nbSamples, nbChannels, ~] = size(data);

channel_average = mean(data(:, :, :), 3);
hamming_channel = repmat(hamming(nbSamples), 1, nbChannels) .* channel_average;
fft_hamming_channel = abs(fft(hamming_channel));
channel_average_frequency = fft_hamming_channel(1:size(fft_hamming_channel, 1)/2, :);

frequency_vector = linspace(0, sampling_frequency/2, size(fft_hamming_channel, 1)/2);

end

