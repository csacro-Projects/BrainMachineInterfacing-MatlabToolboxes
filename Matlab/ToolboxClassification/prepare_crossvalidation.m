function [partitions_class1, partitions_class2, random_idx_class1, random_idx_class2, random_idx_class1_stable, random_idx_class2_stable] = prepare_crossvalidation(nbFolds, nbMatchesClass1, nbMatchesClass2)
%prepare_crossvalidation
%retruns the partitions and random indices for cross validation
%   nbFolds: number of folds for the cross validation
%   nbMatchesClass1: number of samples for class1
%   nbMatchesClass2: number of samples for class2

rest_class1 = mod(nbMatchesClass1, nbFolds);
rest_class2 = mod(nbMatchesClass2, nbFolds);
partitions_class1 = ( (nbMatchesClass1-rest_class1) / nbFolds) * ones(nbFolds, 1);
partitions_class1(1:rest_class1) = partitions_class1(1:rest_class1) +1;
partitions_class2 = ( (nbMatchesClass2-rest_class2) / nbFolds) * ones(nbFolds, 1);
partitions_class2(1:rest_class2) = partitions_class2(1:rest_class2) +1;

random_idx_class1 = randperm(nbMatchesClass1)';
random_idx_class1_stable = random_idx_class1;
random_idx_class2 = randperm(nbMatchesClass2)';
random_idx_class2_stable = random_idx_class2;

end

