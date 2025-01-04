function [] = plot_channelAvg_time(time_vector, channel_average_time, nbPlotRows, yLim)
%plot_channelAvg_time
%plots the coherent average across all trials for each channel in the time domain
%   time_vector: time_vector for plotting
%   channel_average_time: size(channel_average_time) = [nbSamples, nbChannels, (nbData)]
%   nbPlotRows: number of rows in the plot with nbChannels%nbPlotRows=0
%   yLim (optional): limit for the amplitude, e.g. [0, 50]

if size(size(channel_average_time), 2) == 3
    [~, nbChannels, nbData] = size(channel_average_time);
elseif size(size(channel_average_time), 2) == 2
    nbData = 1;
    [~, nbChannels] = size(channel_average_time);
else
    error("channel_average_time does not have 2 or 3 dimensions") 
end

if mod(nbChannels, nbPlotRows) ~= 0
    error("nbChannels%nbPlotColumns!=0")
end

figure('Name', 'Coherent Average across All Trials for')
for i = 1:nbChannels
    subplot(nbPlotRows, nbChannels/nbPlotRows, i)
    if nbData > 1
        hold on;
        for j = 1:nbData
            plot(time_vector, channel_average_time(:, i, j));
        end
        hold off;
    else
        plot(time_vector, channel_average_time(:, i));
    end
    title(['Channel ', num2str(i)])
    ylabel('Amplitude')
    if nargin == 4
        ylim(yLim)
    end
    xlabel('Time [s]')
end

end

