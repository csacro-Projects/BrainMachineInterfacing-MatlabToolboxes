%% before execution
clc; close all; clear;
rng(17) % fix seed

addpath(strcat(pwd,'\ToolboxFeatureExtraction'));
addpath(strcat(pwd,'\ToolboxClassification'));

%% prepare everything (mainly data)
% Donchin matrix
nbTargets=12;
donchin = [
    'A' 'B' 'C' 'D' 'E' 'F';
    'G' 'H' 'I' 'J' 'K' 'L';
    'M' 'N' 'O' 'P' 'Q' 'R';
    'S' 'T' 'U' 'V' 'W' 'X';
    'Y' 'Z' '1' '2' '3' '4';
    '5' '6' '7' '8' '9' '_'];

% load data
data = importdata('datasets/P300data.mat');
data_training = data.training;
data_testing = data.testing;

nbRepeats = 10;
nbColumns = 11;
CHANNEL_COLUMNS = 1:8;
nbChannels = 8;
TIME_COLUMN = 9;
STIMULUS_COLUMN = 10;
LABEL_COLUMN = 11;

% split data into trials
stimulus_onsets = [1; find(data_training(:, STIMULUS_COLUMN)); size(data_training, 1)+1];
stimulus_onsets_training = diff(stimulus_onsets);
stimulus_onsets_training = stimulus_onsets_training(stimulus_onsets_training~=0); % avoid empty (first) cell
data_training_trials = mat2cell(data_training, stimulus_onsets_training);

stimulus_onsets = [1; find(data_testing(:, STIMULUS_COLUMN)); size(data_testing, 1)+1];
stimulus_onsets_testing = diff(stimulus_onsets);
stimulus_onsets_testing = stimulus_onsets_testing(stimulus_onsets_testing~=0); % avoid empty (first) cell
data_testing_trials = mat2cell(data_testing, stimulus_onsets_testing);

% potentially delete first cell (check if this has already been a trial)
if  data_training_trials{1}(1, STIMULUS_COLUMN) == 0
    data_training_trials(1) = [];
end
if  data_testing_trials{1}(1, STIMULUS_COLUMN) == 0
    data_testing_trials(1) = [];
end

% set time vector to start from 0 for every trial
for i = 1:size(data_training_trials, 1)
    init_time = data_training_trials{i}(1, TIME_COLUMN);
    data_training_trials{i}(:, TIME_COLUMN) = data_training_trials{i}(:, TIME_COLUMN) - init_time;
end
for i = 1:size(data_testing_trials, 1)
    init_time = data_testing_trials{i}(1, TIME_COLUMN);
    data_testing_trials{i}(:, TIME_COLUMN) = data_testing_trials{i}(:, TIME_COLUMN) - init_time;
end

% zero-padding such that all trials have the same length
nbSamples = median(cellfun(@(x)size(x,1),data_training_trials));
if mod(nbSamples, 2) ~= 0 % nicer value for extracting features
    nbSamples = nbSamples + 1;
end

data_training_too_long = cellfun(@(x)size(x,1) > nbSamples, data_training_trials);
data_training_too_short = cellfun(@(x)size(x,1) < nbSamples, data_training_trials);
data_training_trials(data_training_too_long) = cellfun(@(x)x(1:nbSamples, :), data_training_trials(data_training_too_long), 'un', 0);
data_training_trials(data_training_too_short) = cellfun(@(x)[x; zeros(nbSamples-size(x,1), nbColumns)], data_training_trials(data_training_too_short), 'un', 0);

data_testing_too_long = cellfun(@(x)size(x,1) > nbSamples, data_testing_trials);
data_testing_too_short = cellfun(@(x)size(x,1) < nbSamples, data_testing_trials);
data_testing_trials(data_testing_too_long) = cellfun(@(x)x(1:nbSamples, :), data_testing_trials(data_testing_too_long), 'un', 0);
data_testing_trials(data_testing_too_short) = cellfun(@(x)[x; zeros(nbSamples-size(x,1), nbColumns-1)], data_testing_trials(data_testing_too_short), 'un', 0);

% split training data into frequent and infrequent responses
data_training_frequent_trials = [];
data_training_infrequent_trials = [];
for i = 1:size(data_training_trials, 1)
    if data_training_trials{i}(1, LABEL_COLUMN) == 1
         data_training_infrequent_trials = vertcat(data_training_infrequent_trials, data_training_trials(i));
     else
         data_training_frequent_trials = vertcat(data_training_frequent_trials, data_training_trials(i));
     end
 end

% transform cell array back to matrix, size() = [nbSamples*nbTrials, nbColumns]
data_training_matrix = cell2mat(data_training_trials);
data_testing_matrix = cell2mat(data_testing_trials);

data_training_frequent_matrix = cell2mat(data_training_frequent_trials);
data_training_infrequent_matrix = cell2mat(data_training_infrequent_trials);

% group repetitions together such that size() = [nbSamples, nbTargets, nbRepeats, nbLetters, nbColumns]
data_training_matrix = reshape(data_training_matrix, nbSamples, nbTargets, nbRepeats, [], nbColumns);
data_testing_matrix = reshape(data_testing_matrix, nbSamples, nbTargets, nbRepeats, [], nbColumns-1);

data_training_frequent_matrix = reshape(data_training_frequent_matrix, nbSamples, nbTargets-2, nbRepeats, [], nbColumns);
data_training_infrequent_matrix = reshape(data_training_infrequent_matrix, nbSamples, 2, nbRepeats, [], nbColumns);

% order nbTargets column from 1 to 12 (for meaningful indexing)
for i = 1:size(data_training_matrix,3)
    for j = 1:size(data_training_matrix,4)
        [~, sort_idx] = sortrows(squeeze(data_training_matrix(1, :, i, j, :)), STIMULUS_COLUMN);
        for k = 1:size(data_training_matrix,1)
            data_training_matrix(k, :, i, j, :) = data_training_matrix(k, sort_idx, i, j, :);
        end
    end
end
for i = 1:size(data_testing_matrix,3)
    for j = 1:size(data_testing_matrix,4)
        [~, sort_idx] = sortrows(squeeze(data_testing_matrix(1, :, i, j, :)), STIMULUS_COLUMN);
        for k = 1:size(data_testing_matrix,1)
            data_testing_matrix(k, :, i, j, :) = data_testing_matrix(k, sort_idx, i, j, :);
        end
    end
end
for i = 1:size(data_training_frequent_matrix,3)
    for j = 1:size(data_training_frequent_matrix,4)
        [~, sort_idx] = sortrows(squeeze(data_training_frequent_matrix(1, :, i, j, :)), STIMULUS_COLUMN);
        for k = 1:size(data_training_frequent_matrix,1)
            data_training_frequent_matrix(k, :, i, j, :) = data_training_frequent_matrix(k, sort_idx, i, j, :);
        end
    end
end
for i = 1:size(data_training_infrequent_matrix,3)
    for j = 1:size(data_training_infrequent_matrix,4)
        [~, sort_idx] = sortrows(squeeze(data_training_infrequent_matrix(1, :, i, j, :)), STIMULUS_COLUMN);
        for k = 1:size(data_training_infrequent_matrix,1)
            data_training_infrequent_matrix(k, :, i, j, :) = data_training_infrequent_matrix(k, sort_idx, i, j, :);
        end
    end
end

% rearrange matrix to have size() = [nbSamples, nbColumns, nbMatches = nbTargets * nbLetters, nbRepeats]
data_training_matrix = permute(data_training_matrix, [1, 5, 2, 4, 3]);
data_testing_matrix = permute(data_testing_matrix, [1, 5, 2, 4, 3]);

data_training_frequent_matrix = permute(data_training_frequent_matrix, [1, 5, 2, 4, 3]);
data_frequent = reshape(data_training_frequent_matrix, nbSamples, nbColumns, [], nbRepeats);
data_training_infrequent_matrix = permute(data_training_infrequent_matrix, [1, 5, 2, 4, 3]);
data_infrequent = reshape(data_training_infrequent_matrix, nbSamples, nbColumns, [], nbRepeats);
nbMatchesFrequent = size(data_frequent, 3);
nbMatchesInfrequent = size(data_infrequent, 3);

%% extract features for Fisher Discriminant Analysis
sampling_frequency = find(data_training(:, TIME_COLUMN, :) < 1, 1, 'last' );

% average over matches and filter out delta band
data_infrequent_avg = squeeze(mean(data_infrequent(:, CHANNEL_COLUMNS, :, :), 3));
data_frequent_avg = squeeze(mean(data_frequent(:, CHANNEL_COLUMNS, :, :), 3));
[b, a] = butter(3, [0.5*(2/sampling_frequency), 4*(2/sampling_frequency)]); % delta band
data_infrequent_avg_filtered = filter(b,a,data_infrequent_avg);
data_frequent_avg_filtered = filter(b,a,data_frequent_avg);

% ERP (time-domain) per channel
[time_vector, channel_average_time_infrequent] = calc_channelAvg_time(data_infrequent_avg_filtered, sampling_frequency);
[~, channel_average_time_frequent] = calc_channelAvg_time(data_frequent_avg_filtered, sampling_frequency);
channel_average_time = zeros([size(channel_average_time_infrequent), 2]);
channel_average_time(:, :, 1) = channel_average_time_infrequent;
channel_average_time(:, :, 2) = channel_average_time_frequent;
plot_channelAvg_time(time_vector, channel_average_time, 2);

% Periodogram (frequency-domain) per channel 
[frequency_vector, channel_average_frequency_infrequent] = calc_channelAvg_frequency(data_infrequent_avg, sampling_frequency);
[~, channel_average_frequency_frequent] = calc_channelAvg_frequency(data_frequent_avg, sampling_frequency);
channel_average_frequency = zeros([size(channel_average_frequency_infrequent), 2]);
channel_average_frequency(:, :, 1) = channel_average_frequency_infrequent;
channel_average_frequency(:, :, 2) = channel_average_frequency_frequent;
plot_channelAvg_frequency(frequency_vector, channel_average_frequency, 2);

% Spectrogram (time-frequency-domain) per channel
[time_vector_spectrogram, frequency_vector_spectrogram, channel_average_spectrogram_infrequent] = calc_channelAvg_spectrogram(data_infrequent_avg, sampling_frequency, 32, 2);
[~, ~, channel_average_spectrogram_frequent] = calc_channelAvg_spectrogram(data_frequent_avg, sampling_frequency, 32, 2);
colorLim = squeeze([
    min(min(min([channel_average_spectrogram_infrequent; channel_average_spectrogram_frequent])))...
    max(max(max([channel_average_spectrogram_infrequent; channel_average_spectrogram_frequent])))
    ]);
plot_channelAvg_spectrogram(time_vector_spectrogram, frequency_vector_spectrogram, channel_average_spectrogram_infrequent, nbChannels, [0, 50], colorLim, 2);
plot_channelAvg_spectrogram(time_vector_spectrogram, frequency_vector_spectrogram, channel_average_spectrogram_frequent, nbChannels, [0, 50], colorLim, 2);

% correlation in time and frequency domain per channel
r2_time = calc_correlation(data_infrequent_avg, data_frequent_avg);
plot_correlation_selected(r2_time);
[~, channel_frequency_infrequent] = calc_channel_frequency(data_infrequent_avg, sampling_frequency);
[~, channel_frequency_frequent] = calc_channel_frequency(data_frequent_avg, sampling_frequency);
r2_frequency = calc_correlation(channel_frequency_infrequent, channel_frequency_frequent);
plot_correlation_selected(r2_frequency);

%% Fisher Discriminant Analysis
% 5-fold cross validation
nbFolds = 5;
[partitions_infrequent, partitions_frequent, random_idx_infrequent, random_idx_frequent, random_idx_infrequent_stable, random_idx_frequent_stable] = ...
    prepare_crossvalidation(nbFolds, nbMatchesInfrequent, nbMatchesFrequent);
accuracy_crossval = zeros(nbFolds, 1);
for fold=1:nbFolds
    [infrequent_val, infrequent_train, frequent_val, frequent_train, random_idx_infrequent, random_idx_frequent] = ...
        splitdata_crossvalidation(fold, data_infrequent(:, CHANNEL_COLUMNS, :, :), data_frequent(:, CHANNEL_COLUMNS, :, :),...
        partitions_infrequent, partitions_frequent, random_idx_infrequent, random_idx_frequent, random_idx_infrequent_stable, random_idx_frequent_stable);
    % train and eval Fisher
    [feature, label] = prepare_fisher_discriminant_analysis(infrequent_train, frequent_train, @P300Speller_createFeature);
    model_fisher = train_fisher_discriminant_analysis(feature, label);
    eval_fisher_discriminant_analysis(model_fisher, feature, label);
    [feature, label] = prepare_fisher_discriminant_analysis(infrequent_val, frequent_val, @P300Speller_createFeature);
    [~, accuracy] = eval_fisher_discriminant_analysis(model_fisher, feature, label);
    accuracy_crossval(fold) = accuracy; 
end
disp("Result of 5-fold cross validation")
disp("Fisher Discriminant Analysis - accuracy: mean " + mean(accuracy_crossval) + " variance " + std(accuracy_crossval))

% train classifier on complete training data
[feature, label] = prepare_fisher_discriminant_analysis(data_infrequent(:, CHANNEL_COLUMNS, :, :), data_frequent(:, CHANNEL_COLUMNS, :, :), @P300Speller_createFeature);
model_fisher = train_fisher_discriminant_analysis(feature, label);
[~, accuracy] = eval_fisher_discriminant_analysis(model_fisher, feature, label);
plot_fisher_discriminant_analysis(model_fisher, feature, label);
disp("Result of training on the complete training data")
disp("Fisher Discriminant Analysis - accuracy on training data:" + accuracy)

%% Stepwise Regression
% 5-fold cross validation
nbFolds = 5;
[partitions_infrequent, partitions_frequent, random_idx_infrequent, random_idx_frequent, random_idx_infrequent_stable, random_idx_frequent_stable] = ...
    prepare_crossvalidation(nbFolds, nbMatchesInfrequent, nbMatchesFrequent);
accuracy_crossval = zeros(nbFolds, 1);
for fold=1:nbFolds
    [infrequent_val, infrequent_train, frequent_val, frequent_train, random_idx_infrequent, random_idx_frequent] = ...
        splitdata_crossvalidation(fold, data_infrequent(:, CHANNEL_COLUMNS, :, :), data_frequent(:, CHANNEL_COLUMNS, :, :),...
        partitions_infrequent, partitions_frequent, random_idx_infrequent, random_idx_frequent, random_idx_infrequent_stable, random_idx_frequent_stable);
    % train and eval Regression
    [data, label] = prepare_stepwise_regression(squeeze(mean(infrequent_train, 4)), squeeze(mean(frequent_train, 4)));
    [coefficients, b0, finalmodel, pval] = train_stepwise_regression(data, label);
    eval_stepwise_regression(coefficients, b0, finalmodel, pval, data, label);
    [data, label] = prepare_stepwise_regression(squeeze(mean(infrequent_val, 4)), squeeze(mean(frequent_val, 4)));
    [~, accuracy] = eval_stepwise_regression(coefficients, b0, finalmodel, pval, data, label);
    accuracy_crossval(fold) = accuracy; 
end
disp("Result of 5-fold cross validation")
disp("Stepwise Regression - accuracy: mean " + mean(accuracy_crossval) + " variance " + std(accuracy_crossval))

% train classifier on complete training data
[data, label] = prepare_stepwise_regression(squeeze(mean(data_infrequent(:, CHANNEL_COLUMNS, :, :), 4)), squeeze(mean(data_frequent(:, CHANNEL_COLUMNS, :, :), 4)));
[coefficients, b0, finalmodel, pval] = train_stepwise_regression(data, label);
[~, accuracy] = eval_stepwise_regression(coefficients, b0, finalmodel, pval, data, label);
plot_stepwise_regression(coefficients, b0, finalmodel, pval, data, label);
disp("Result of training on the complete training data")
disp("Stepwise Regression - accuracy on training data:" + accuracy)

%% apply better performing classifier (Stepwise Linear Regression) to train and test data to retrieve the spelled words
word_training = [];
for letter = 1:size(data_training_matrix, 4)
    data_training_regression = prepare_data_stepwise_regression(squeeze(mean(data_training_matrix(:, CHANNEL_COLUMNS, :, letter, :), 5)));
    %data_training_fisher = P300Speller_createFeature(data_training_matrix(:, CHANNEL_COLUMNS, :, letter, :));
    
    [row_prediction, row_distance] = predict_stepwise_regression(coefficients, b0, finalmodel, data_training_regression(1:6, :));
    %[row_prediction, row_distance] = predict_fisher_discriminant_analysis(model_fisher, data_training_fisher(1:6, :));
    [~, infrequentRow_pos] = sort(row_distance, 'ascend'); % the smaller distance the likelier infrequent
    infrequentRow = infrequentRow_pos(1);
    
    [col_prediction, col_distance] = predict_stepwise_regression(coefficients, b0, finalmodel, data_training_regression(7:12, :));
    %[col_prediction, col_distance] = predict_fisher_discriminant_analysis(model_fisher, data_training_fisher(7:12, :));
    [~, infrequentCol_pos] = sort(col_distance, 'ascend'); % the smaller distance the likelier infrequent
    infrequentCol = infrequentCol_pos(1);
    
    word_training = strcat(word_training, donchin(infrequentRow,infrequentCol));
end
disp("The spelled word in the training is: " + word_training)

word_testing = [];
for letter = 1:size(data_testing_matrix, 4)
    data_testing_regression = prepare_data_stepwise_regression(squeeze(mean(data_testing_matrix(:, CHANNEL_COLUMNS, :, letter, :), 5)));
    %data_testing_fisher = P300Speller_createFeature(data_testing_matrix(:, CHANNEL_COLUMNS, :, letter, :));
    
    [row_prediction, row_distance] = predict_stepwise_regression(coefficients, b0, finalmodel, data_testing_regression(1:6,:));
    %[row_prediction, row_distance] = predict_fisher_discriminant_analysis(model_fisher, data_testing_fisher(1:6, :));
    [~, infrequentRow_pos] = sort(row_distance, 'ascend'); % the smaller distance the likelier infrequent
    infrequentRow = infrequentRow_pos(1);
    
    [col_prediction, col_distance] = predict_stepwise_regression(coefficients, b0, finalmodel, data_testing_regression(7:12,:));
    %[col_prediction, col_distance] = predict_fisher_discriminant_analysis(model_fisher, data_testing_fisher(7:12, :));
    [~, infrequentCol_pos] = sort(col_distance, 'ascend'); % the smaller distance the likelier infrequent
    infrequentCol = infrequentCol_pos(1);
    
    word_testing = strcat(word_testing, donchin(infrequentRow,infrequentCol));
end
disp("The spelled word in the testing is: " + word_testing)

%% after execution
rmpath(strcat(pwd,'\ToolboxFeatureExtraction'));
rmpath(strcat(pwd,'\ToolboxClassification'));
