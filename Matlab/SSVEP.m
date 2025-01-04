%% before execution
clc; close all; clear;
rng(17) % fix seed

addpath(strcat(pwd,'\ToolboxFeatureExtraction'));
addpath(strcat(pwd,'\ToolboxClassification'));

%% prepare everything (mainly data)
% load data
data = importdata('datasets/SSVEP.mat');
data_signal = data.signal;
data_cue = data.cue;

nbChannels = 12;
nbTrials = 10;
sampling_frequency = 256;
stimulation_time = 15;
nbSamples = stimulation_time * sampling_frequency;

% extract stimulations periods and split into trials such that size() = [nbSamples, nbChannels, nbTrials]
stimulation_periods = find(data_cue == 1);
data_signal_stimulation = data_signal(stimulation_periods, :);
data_signal_stimulation_trials = reshape(data_signal_stimulation, [nbSamples, nbTrials, nbChannels]);
data_signal_stimulation_trials = permute(data_signal_stimulation_trials, [1, 3, 2]);

%% determine freq_slow and freq_fast
% exclude first second of each trial (we assume that this second mainly refelcts reactions due to stimulus onset)
data_signal_stimulation_trials = data_signal_stimulation_trials(sampling_frequency+1:end, :, :);

% filter out relevant frequency band
[b, a] = butter(2, [5*(2/sampling_frequency), 45*(2/sampling_frequency)]); % 5Hz to 45Hz
data_signal_stimulation_trials_filtered = filter(b,a,data_signal_stimulation_trials);

% plot power spectrum
[frequency_vector, channel_average_frequency] = calc_channelAvg_frequency(data_signal_stimulation_trials_filtered, sampling_frequency);
channel_average_frequency = mean(channel_average_frequency, 2); % average out noise
plot_channelAvg_frequency(frequency_vector, channel_average_frequency, 1);

% deterimine frequencies with peaks (that are not harmonics)
[~, idx_max1] = max(channel_average_frequency);
max1_frequency = frequency_vector(idx_max1)
[~, idx_max2] = max(channel_average_frequency(channel_average_frequency<max(channel_average_frequency)));
max2_frequency = frequency_vector(idx_max2)

freq_low = 8;
freq_fast = 14;

%% determine the sequence of yes/freq_low and no/freq_fast
[frequency_vector, channel_frequency] = calc_channel_frequency(data_signal_stimulation_trials_filtered, sampling_frequency);
channel_frequency = mean(channel_frequency, 2); % average out noise
channel_frequency = abs(channel_frequency).^2; % power spectrum
for trial = 1:nbTrials
    channel_frequency_trial = channel_frequency(:, :, trial);
    plot_channelAvg_frequency(frequency_vector, channel_frequency_trial, 1);
    [~, idx_max] = max(channel_frequency_trial);
    freq = frequency_vector(idx_max);
    if abs(freq_fast - freq) < abs(freq_low - freq)
        % freq is closer to freq_fast -> No
        disp("Trial " + trial + ": No")
    else
        disp("Trial " + trial + ": Yes")
    end    
end

%% after execution
rmpath(strcat(pwd,'\ToolboxFeatureExtraction'));
rmpath(strcat(pwd,'\ToolboxClassification'));