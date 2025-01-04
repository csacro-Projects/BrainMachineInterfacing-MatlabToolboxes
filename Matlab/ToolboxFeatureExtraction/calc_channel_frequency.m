function [frequency_vector, channel_frequency] = calc_channel_frequency(data, sampling_frequency)
%calc_channelAvg_frequency
%returns the frequency_vector and channel_frequency with size(channel_frequency) = [nbSamples/2, nbChannels, nbTrials]
%   data: recorded signals with size(data) = [nbSamples, nbChannels, nbTrials]
%   sampling_frequency: frequency at which the signals were sampled

if size(size(data), 2) ~= 3
    error("data does not have 3 dimensions")
end
[nbSamples, nbChannels, nbTrials] = size(data);

hamming_channel = repmat(hamming(nbSamples), [1, nbChannels, nbTrials]) .* data;
fft_hamming_channel = abs(fft(hamming_channel));
channel_frequency = fft_hamming_channel(1:size(fft_hamming_channel, 1)/2, :, :);

frequency_vector = linspace(0, sampling_frequency/2, size(fft_hamming_channel, 1)/2);

end

