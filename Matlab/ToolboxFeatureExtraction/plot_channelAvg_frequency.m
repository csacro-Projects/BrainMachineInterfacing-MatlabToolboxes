function [] = plot_channelAvg_frequency(frequency_vector, channel_average_frequency, nbPlotRows)
%plot_channelAvg_frequency
%plots the coherent average across all trials for each channel in the frequency domain
%   frequency_vector: frequency_vector for plotting
%   channel_average_frequency: size(channel_average_frequency) = [nbSamples/2, nbChannels, (nbData)]
%   nbPlotRows: number of rows in the plot with nbChannels%nbPlotRows=0

if size(size(channel_average_frequency), 2) == 3
    [~, nbChannels, nbData] = size(channel_average_frequency);
elseif size(size(channel_average_frequency), 2) == 2
    nbData = 1;
    [~, nbChannels] = size(channel_average_frequency);
else
   error("channel_average_frequency does not have 2 or 3 dimensions") 
end

if mod(nbChannels, nbPlotRows) ~= 0
    error("nbChannels%nbPlotColumns!=0")
end

figure('Name', 'Single-sided Spectrum for')
for i = 1:nbChannels
    subplot(nbPlotRows, nbChannels/nbPlotRows, i)
    if nbData > 1
        hold on;
        for j = 1:nbData
            plot(frequency_vector, channel_average_frequency(:, i, j));
        end
        hold off;
    else 
        plot(frequency_vector, channel_average_frequency(:, i));
    end
    title(['Channel ', num2str(i)])
    ylabel('Amplitude')
    xlabel('Frequency [Hz]')
end

end



