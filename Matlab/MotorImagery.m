%% before execution
clc; close all; clear;
rng(17) % fix seed

addpath(strcat(pwd,'\ToolboxFeatureExtraction'));
addpath(strcat(pwd,'\ToolboxClassification'));

%% prepare everything (mainly data)
% load data
data = importdata('datasets/MIdata.mat');
data_signal = data.signal;
data_timeVector = data.timeVector;
data_labels = data.labels;

nbChannels = 26;

IDLE = 0;
LEFT_HAND = 1;
RIGHT_HAND = 2;
BOTH_HANDS = 3;
BOTH_FEET = 4;
UNLABELED = -1;
data_labels(isnan(data_labels)) = UNLABELED; % better to work with numbers than with NaN
sampling_frequency = find(data_timeVector < 1, 1, 'last' ); % we will work with this instead of data_timeVector

% split data into trials and classes such that size() = [nbSamples, nbChannels, nbMatches]
data_training_rightHand = [];
data_training_leftHand = [];
data_training_bothHands = [];
data_training_bothFeet = [];
data_testing = [];

trial_start_index = 0;
trial_label = 0;
for i = 1:length(data_labels)
    label = data_labels(i);
    
    if trial_label == label
        % we are still in the same trial
        continue;
    end
    if trial_label ~= IDLE
        % we have finished a trial
        switch trial_label
            case LEFT_HAND
                data_training_leftHand = cat(3, data_training_leftHand, data_signal(trial_start_index:i-1, :));
            case RIGHT_HAND
                data_training_rightHand = cat(3, data_training_rightHand, data_signal(trial_start_index:i-1, :));
            case BOTH_HANDS
                data_training_bothHands = cat(3, data_training_bothHands, data_signal(trial_start_index:i-1, :));
            case BOTH_FEET
                data_training_bothFeet = cat(3, data_training_bothFeet, data_signal(trial_start_index:i-1, :));
            case UNLABELED
                data_testing = cat(3, data_testing, data_signal(trial_start_index:i-1, :));
        end
    end
    trial_start_index = i;
    trial_label = label;          
end

% (in case we would want to use hierarchical multi-class approach instead of OVO)
data_training_both = cat(3, data_training_bothHands, data_training_bothFeet);
data_training_both = data_training_both(:, :, randperm(size(data_training_both, 3))); % just to be sure

data_training_single = cat(3, data_training_leftHand, data_training_rightHand);
data_training_single = data_training_single(:, :, randperm(size(data_training_single, 3))); % just to be sure

% filter out alpha- and beta-band and remove first second of recording
[b, a] = butter(5, [8*(2/sampling_frequency), 30*(2/sampling_frequency)]);

data_training_leftHand_filtered = filter(b,a,data_training_leftHand(sampling_frequency+1:end, :, :));
data_training_rightHand_filtered = filter(b,a,data_training_rightHand(sampling_frequency+1:end, :, :));
data_training_bothHands_filtered = filter(b,a,data_training_bothHands(sampling_frequency+1:end, :, :));
data_training_bothFeet_filtered = filter(b,a,data_training_bothFeet(sampling_frequency+1:end, :, :));

data_testing_filtered = filter(b,a,data_testing(sampling_frequency+1:end, :, :));

data_training_both_filtered = filter(b,a,data_training_both(sampling_frequency+1:end, :, :));
data_training_single_filtered = filter(b,a,data_training_single(sampling_frequency+1:end, :, :));

%% extract custom features for the model without CSP
% Timeseries (time-domain) per channel
[timeseries_vector, channel_timeseries_leftHand] = calc_channel_timeseries(data_training_leftHand_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_rightHand] = calc_channel_timeseries(data_training_rightHand_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_bothHands] = calc_channel_timeseries(data_training_bothHands_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_bothFeet] = calc_channel_timeseries(data_training_bothFeet_filtered, sampling_frequency, 512, 4);
channel_average_timeseries_leftHand = mean(channel_timeseries_leftHand(:, :, :), 3);
channel_average_timeseries_rightHand = mean(channel_timeseries_rightHand(:, :, :), 3);
channel_average_timeseries_bothHands = mean(channel_timeseries_bothHands(:, :, :), 3);
channel_average_timeseries_bothFeet = mean(channel_timeseries_bothFeet(:, :, :), 3);
channel_average_timeseries = zeros([size(channel_average_timeseries_leftHand), 4]);
channel_average_timeseries(:, :, 1) = channel_average_timeseries_leftHand;
channel_average_timeseries(:, :, 2) = channel_average_timeseries_rightHand;
channel_average_timeseries(:, :, 3) = channel_average_timeseries_bothHands;
channel_average_timeseries(:, :, 4) = channel_average_timeseries_bothFeet;
plot_channelAvg_time(timeseries_vector, channel_average_timeseries, 2, [0, 50]);

% Periodograms (frequency-domain) per channel 
[frequency_vector, channel_frequency_leftHand] = calc_channel_frequency(data_training_leftHand_filtered, sampling_frequency);
[~, channel_frequency_bothHands] = calc_channel_frequency(data_training_rightHand_filtered, sampling_frequency);
[~, channel_frequency_bothFeet] = calc_channel_frequency(data_training_bothHands_filtered, sampling_frequency);
[~, channel_frequency_rightHand] = calc_channel_frequency(data_training_bothFeet_filtered, sampling_frequency);
channel_average_frequency_leftHand = mean(channel_frequency_leftHand, 3);
channel_average_frequency_rightHand = mean(channel_frequency_rightHand, 3);
channel_average_frequency_bothHands = mean(channel_frequency_bothHands, 3);
channel_average_frequency_bothFeet = mean(channel_frequency_bothFeet, 3);
channel_average_frequency = zeros([size(channel_average_frequency_leftHand), 4]);
channel_average_frequency(:, :, 1) = channel_average_frequency_leftHand;
channel_average_frequency(:, :, 2) = channel_average_frequency_rightHand;
channel_average_frequency(:, :, 3) = channel_average_frequency_bothHands;
channel_average_frequency(:, :, 4) = channel_average_frequency_bothFeet;
plot_channelAvg_frequency(frequency_vector, channel_average_frequency, 2);

% correlation in time and frequency domain per channel and per binary classification
% bothHandsVSbothFeed
[~, channel_frequency_class1] = calc_channel_frequency(data_training_bothHands_filtered, sampling_frequency);
[~, channel_frequency_class2] = calc_channel_frequency(data_training_bothFeet_filtered, sampling_frequency);
r2_frequency = calc_correlation(channel_frequency_class1, channel_frequency_class2);
plot_correlation_selected(r2_frequency);
[~, channel_timeseries_class1] = calc_channel_timeseries(data_training_bothHands_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_class2] = calc_channel_timeseries(data_training_bothFeet_filtered, sampling_frequency, 512, 4);
r2_frequency = calc_correlation(channel_timeseries_class1, channel_timeseries_class2);
plot_correlation_selected(r2_frequency);
% bothHandsVSleftHand
[~, channel_frequency_class1] = calc_channel_frequency(data_training_bothHands_filtered, sampling_frequency);
[~, channel_frequency_class2] = calc_channel_frequency(data_training_leftHand_filtered, sampling_frequency);
r2_frequency = calc_correlation(channel_frequency_class1, channel_frequency_class2);
plot_correlation_selected(r2_frequency);
[~, channel_timeseries_class1] = calc_channel_timeseries(data_training_bothHands_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_class2] = calc_channel_timeseries(data_training_leftHand_filtered, sampling_frequency, 512, 4);
r2_frequency = calc_correlation(channel_timeseries_class1, channel_timeseries_class2);
plot_correlation_selected(r2_frequency);
% bothHandsVSrightHand
[~, channel_frequency_class1] = calc_channel_frequency(data_training_bothHands_filtered, sampling_frequency);
[~, channel_frequency_class2] = calc_channel_frequency(data_training_rightHand_filtered, sampling_frequency);
r2_frequency = calc_correlation(channel_frequency_class1, channel_frequency_class2);
plot_correlation_selected(r2_frequency);
[~, channel_timeseries_class1] = calc_channel_timeseries(data_training_bothHands_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_class2] = calc_channel_timeseries(data_training_rightHand_filtered, sampling_frequency, 512, 4);
r2_frequency = calc_correlation(channel_timeseries_class1, channel_timeseries_class2);
plot_correlation_selected(r2_frequency);
% bothFeetVSleftHand
[~, channel_frequency_class1] = calc_channel_frequency(data_training_bothFeet_filtered, sampling_frequency);
[~, channel_frequency_class2] = calc_channel_frequency(data_training_leftHand_filtered, sampling_frequency);
r2_frequency = calc_correlation(channel_frequency_class1, channel_frequency_class2);
plot_correlation_selected(r2_frequency);
[~, channel_timeseries_class1] = calc_channel_timeseries(data_training_bothFeet_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_class2] = calc_channel_timeseries(data_training_leftHand_filtered, sampling_frequency, 512, 4);
r2_frequency = calc_correlation(channel_timeseries_class1, channel_timeseries_class2);
plot_correlation_selected(r2_frequency);
% bothFeetVSrightHand
[~, channel_frequency_class1] = calc_channel_frequency(data_training_bothFeet_filtered, sampling_frequency);
[~, channel_frequency_class2] = calc_channel_frequency(data_training_rightHand_filtered, sampling_frequency);
r2_frequency = calc_correlation(channel_frequency_class1, channel_frequency_class2);
plot_correlation_selected(r2_frequency);
[~, channel_timeseries_class1] = calc_channel_timeseries(data_training_bothFeet_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_class2] = calc_channel_timeseries(data_training_rightHand_filtered, sampling_frequency, 512, 4);
r2_frequency = calc_correlation(channel_timeseries_class1, channel_timeseries_class2);
plot_correlation_selected(r2_frequency);
% leftHandVSrightHand
[~, channel_frequency_class1] = calc_channel_frequency(data_training_leftHand_filtered, sampling_frequency);
[~, channel_frequency_class2] = calc_channel_frequency(data_training_rightHand_filtered, sampling_frequency);
r2_frequency = calc_correlation(channel_frequency_class1, channel_frequency_class2);
plot_correlation_selected(r2_frequency);
[~, channel_timeseries_class1] = calc_channel_timeseries(data_training_leftHand_filtered, sampling_frequency, 512, 4);
[~, channel_timeseries_class2] = calc_channel_timeseries(data_training_rightHand_filtered, sampling_frequency, 512, 4);
r2_frequency = calc_correlation(channel_timeseries_class1, channel_timeseries_class2);
plot_correlation_selected(r2_frequency);

%% train an OVO multi-class model with custom features (without CSP)
[model_BHvsBF_custom, model_BHvsLH_custom, model_BHvsRH_custom, model_BFvsLH_custom, model_BFvsRH_custom, model_LHvsRH_custom]...
        = MotorImagery_train_OVO(0, data_training_leftHand_filtered, data_training_rightHand_filtered, data_training_bothHands_filtered, data_training_bothFeet_filtered);

%% train an OVO multi-class model with CSP features
[model_BHvsBF_csp, model_BHvsLH_csp, model_BHvsRH_csp, model_BFvsLH_csp, model_BFvsRH_csp, model_LHvsRH_csp]...
        = MotorImagery_train_OVO(3, data_training_leftHand_filtered, data_training_rightHand_filtered, data_training_bothHands_filtered, data_training_bothFeet_filtered);

%% predict the class labels of the unlabeled trials with better performing OVO multi-class model (with custom features)
result = MotorImagery_predict_OVO(data_testing_filtered, 0, model_BHvsBF_custom, model_BHvsLH_custom, model_BHvsRH_custom,...
                                    model_BFvsLH_custom, model_BFvsRH_custom, model_LHvsRH_custom);
% result = MotorImagery_predict_OVO(data_testing_filtered, 3, model_BHvsBF_csp, model_BHvsLH_csp, model_BHvsRH_csp,...
%                                    model_BFvsLH_csp, model_BFvsRH_csp, model_LHvsRH_csp);                                
result_label = strings(size(result));
result_label(result==LEFT_HAND) = "LEFT_HAND";
result_label(result==RIGHT_HAND) = "RIGHT_HAND";
result_label(result==BOTH_HANDS) = "BOTH_HANDS";
result_label(result==BOTH_FEET) = "BOTH_FEET";
disp("Class labels for unlabeled trials are")
disp(result_label)

%% after execution
rmpath(strcat(pwd,'\ToolboxFeatureExtraction'));
rmpath(strcat(pwd,'\ToolboxClassification'));