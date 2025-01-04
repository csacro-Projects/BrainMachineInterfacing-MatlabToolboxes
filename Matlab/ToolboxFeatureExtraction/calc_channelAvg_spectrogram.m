function [time_vector, frequency_vector, channel_average_spectrogram] = calc_channelAvg_spectrogram(data, sampling_frequency, window_size, step_size)
%calc_channelAvg_spectrogram
%returns the time_vector, frequency_vector and channel_average_spectrogram with size(channel_average_spectrogram) = [nbSteps, window_size/2, nbChannels]
%   data: recorded signals with size(data) = [nbSamples, nbChannels, nbTrials]
%   sampling_frequency: frequency at which the signals were sampled
%   window_size: size of the window, needs to be a power of two value
%   step_size: size of the steps, needs to be a power of two value

log_window_size = log(window_size)/log(2);
log_step_size = log(step_size)/log(2);
if round(log_window_size) - log_window_size ~= 0 ||  round(log_step_size) - log_step_size ~= 0
    error("window_size and step_size need to be power of two values")
end

if size(size(data), 2) ~= 3
    error("data does not have 3 dimensions")
end    
[nbSamples, nbChannels, ~] = size(data);

channel_average = mean(data(:, :, :), 3);

nbSteps = (nbSamples - window_size + step_size) / step_size;
power_spectra = zeros(nbSteps, window_size/2, nbChannels);
for i = 1:nbSteps
    
    factor = (i-1)*step_size;
    window = channel_average(1+factor:window_size+factor, :);
    window_hamming = repmat(hamming(window_size), 1, nbChannels) .* window;
    fft_window_hamming = fft(window_hamming);
    single_sided = fft_window_hamming(1:size(fft_window_hamming, 1)/2, :);
    power_spectrum = abs(single_sided);
    
    power_spectra(i, :, :) = power_spectrum;
end
channel_average_spectrogram = permute(power_spectra, [2 1 3]);

time_vector = window_size/2:step_size:nbSamples-window_size/2;
time_vector = time_vector ./ sampling_frequency;
frequency_vector = linspace(0, sampling_frequency/2, window_size/2);

end

