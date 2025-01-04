function [class1_val, class1_train, class2_val, class2_train, random_idx_class1, random_idx_class2] = splitdata_crossvalidation(fold, data_class1, data_class2, partitions_class1, partitions_class2, random_idx_class1, random_idx_class2, random_idx_class1_stable, random_idx_class2_stable)
%splitdata_crossvalidation
%returns the training and validation data for each class and updates of the indices
%   fold: number of fold one is in
%   data_class1: data of the first class with size(data_class1) = [nbSamples, nbChannels, nbMatchesClass1, nbTrials]
%   data_class2: data of the second class with size(data_class2) = [nbSamples, nbChannels, nbMatchesClass2, nbTrials]
%   other parameters: partitions and random indices for cross validation

disp("fold: " + fold)

idx_class1_val = random_idx_class1(1:partitions_class1(fold));
idx_class1_train = setdiff(random_idx_class1_stable, idx_class1_val);
idx_class2_val = random_idx_class2(1:partitions_class2(fold));
idx_class2_train = setdiff(random_idx_class2_stable, idx_class2_val);

class1_val = data_class1(:,:,idx_class1_val,:);
class1_train = data_class1(:,:,idx_class1_train,:);
class2_val = data_class2(:,:,idx_class2_val,:);
class2_train = data_class2(:,:,idx_class2_train,:);

random_idx_class1 = random_idx_class1(partitions_class1(fold)+1:end);
random_idx_class2 = random_idx_class2(partitions_class2(fold)+1:end);

end

