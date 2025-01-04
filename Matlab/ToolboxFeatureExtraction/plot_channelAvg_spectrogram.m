function [] = plot_channelAvg_spectrogram(time_vector, frequency_vector, channel_average_spectrogram, nbChannels, frequencyLim, colorLim, nbPlotRows)
%plot_channelAvg_spectrogram
%plots the coherent average across all trials for each channel in a spectrogram
%   time_vector: time_vector for plotting
%   frequency_vector: frequency_vector for plotting
%   channel_average_spectrogram: size(channel_average_spectrogram) = [nbSteps, window_size/2, nbChannels]
%   nbChannels: number of channels
%   frequencyLim: limit for the frequeny axis in Hz, e.g. [0, 20]
%   colorLim: limit for the color bar, e.g. [0, 10000]
%   nbPlotRows: number of rows in the plot with nbChannels%nbPlotRows=0

if mod(nbChannels, nbPlotRows) ~= 0
    error("nbChannels%nbPlotColumns!=0")
end

figure('Name', 'Spectrogram for')
for i = 1:nbChannels
    subplot(nbPlotRows, nbChannels/nbPlotRows, i)
    imagesc(time_vector, frequency_vector, channel_average_spectrogram(:, :, i))
    colorbar
    set(gca,'YDir','normal')
    ylim(frequencyLim)
    caxis(colorLim)
    title(['Channel ', num2str(i)])
end

end

