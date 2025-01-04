function [] = plot_correlation_selected(r2_selected)
%plot_correlation_selected
%plots the correlation per frequency and channel and the mean of the correlation topographically
%   r2_selected: correlation to be plotted with size(r2_selected) = [nbChannels, nbFrequencies]

figure('Name', 'Corrleation between data')
imagesc([], 1:1:size(r2_selected, 1), r2_selected)
colormap(flipud(hot))
colorbar
set(gca,'YDir','normal')

end

