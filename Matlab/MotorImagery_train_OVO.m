function [model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH]...
    = MotorImagery_train_OVO(nbFilters, data_leftHand, data_rightHand, data_bothHands, data_bothFeet)

% 5-fold cross validation
nbFolds = 5;
[partitions_leftHand, partitions_rightHand, random_idx_leftHand, random_idx_rightHand, random_idx_leftHand_stable, random_idx_rightHand_stable] = ...
    prepare_crossvalidation(nbFolds, size(data_leftHand,3), size(data_rightHand,3));
[partitions_bothHands, partitions_bothFeet, random_idx_bothHands, random_idx_bothFeet, random_idx_bothHands_stable, random_idx_bothFeet_stable] = ...
    prepare_crossvalidation(nbFolds, size(data_bothHands,3), size(data_bothFeet,3));
accuracies = zeros(nbFolds + 1, 7); % [fold, classifier]

for fold=1:nbFolds
    [leftHand_val, leftHand_train, rightHand_val, rightHand_train, random_idx_leftHand, random_idx_rightHand] = ...
        splitdata_crossvalidation(fold, data_leftHand, data_rightHand, partitions_leftHand, partitions_rightHand, random_idx_leftHand, random_idx_rightHand, random_idx_leftHand_stable, random_idx_rightHand_stable);
    [bothHands_val, bothHands_train, bothFeet_val, bothFeet_train, random_idx_bothHands, random_idx_bothFeet] = ...
        splitdata_crossvalidation(fold, data_bothHands, data_bothFeet, partitions_bothHands, partitions_bothFeet, random_idx_bothHands, random_idx_bothFeet, random_idx_bothHands_stable, random_idx_bothFeet_stable);
    
    if nbFilters ~= 0
        % CSP features
        disp("model_bothHandVSbothFeet")
        cspMatrix_BHvsBF = calc_csp(bothHands_train, bothFeet_train);
        save('MotorImagery_cspMatrix_BHvsBF.mat', 'cspMatrix_BHvsBF');
        [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(bothHands_train, bothFeet_train, cspMatrix_BHvsBF, nbFilters);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis_csp(bothHands_val, bothFeet_val, cspMatrix_BHvsBF, nbFilters);
        [model_BHvsBF, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 1) = accuracy;
        disp("model_bothHandVSleftHand")
        cspMatrix_BHvsLH = calc_csp(bothHands_train, leftHand_train);
		save('MotorImagery_cspMatrix_BHvsLH.mat', 'cspMatrix_BHvsLH');
        [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(bothHands_train, leftHand_train, cspMatrix_BHvsLH, nbFilters);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis_csp(bothHands_val, leftHand_val, cspMatrix_BHvsLH, nbFilters);
        [model_BHvsLH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 2) = accuracy;
        disp("model_bothHandVSrightHand")
        cspMatrix_BHvsRH = calc_csp(bothHands_train, rightHand_train);
		save('MotorImagery_cspMatrix_BHvsRH.mat', 'cspMatrix_BHvsRH');
        [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(bothHands_train, rightHand_train, cspMatrix_BHvsRH, nbFilters);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis_csp(bothHands_val, rightHand_val, cspMatrix_BHvsRH, nbFilters);
        [model_BHvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 3) = accuracy;

        disp("model_bothFeetVSleftHand")
        cspMatrix_BFvsLH = calc_csp(bothFeet_train, leftHand_train);
		save('MotorImagery_cspMatrix_BFvsLH.mat', 'cspMatrix_BFvsLH');
        [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(bothFeet_train, leftHand_train, cspMatrix_BFvsLH, nbFilters);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis_csp(bothFeet_val, leftHand_val, cspMatrix_BFvsLH, nbFilters);
        [model_BFvsLH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 4) = accuracy;
        disp("model_bothFeetVSrightHand")
        cspMatrix_BFvsRH = calc_csp(bothFeet_train, rightHand_train);
		save('MotorImagery_cspMatrix_BFvsRH.mat', 'cspMatrix_BFvsRH');
        [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(bothFeet_train, rightHand_train, cspMatrix_BFvsRH, nbFilters);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis_csp(bothFeet_val, rightHand_val, cspMatrix_BFvsRH, nbFilters);
        [model_BFvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 5) = accuracy;

        disp("model_leftHandVSrightHand")
        cspMatrix_LHvsRH = calc_csp(leftHand_train, rightHand_train);
		save('MotorImagery_cspMatrix_LHvsRH.mat', 'cspMatrix_LHvsRH');
        [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(leftHand_train, rightHand_train, cspMatrix_LHvsRH, nbFilters);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis_csp(leftHand_val, rightHand_val, cspMatrix_LHvsRH, nbFilters);
        [model_LHvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 6) = accuracy;
    else
        % custom features
        disp("model_bothHandsVSbothFeed")
        [feature_train, label_train] = prepare_fisher_discriminant_analysis(bothHands_train, bothFeet_train, @MotorImagery_createBHvsBFfeature);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis(bothHands_val, bothFeet_val, @MotorImagery_createBHvsBFfeature);
        [model_BHvsBF, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 1) = accuracy;
        disp("model_bothHandsVSleftHand")
        [feature_train, label_train] = prepare_fisher_discriminant_analysis(bothHands_train, leftHand_train, @MotorImagery_createBHvsLHfeature);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis(bothHands_val, leftHand_val, @MotorImagery_createBHvsLHfeature);
        [model_BHvsLH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 2) = accuracy;
        disp("model_bothHandsVSrightHand")
        [feature_train, label_train] = prepare_fisher_discriminant_analysis(bothHands_train, rightHand_train, @MotorImagery_createBHvsRHfeature);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis(bothHands_val, rightHand_val, @MotorImagery_createBHvsRHfeature);
        [model_BHvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 3) = accuracy;

        disp("model_bothFeetVSleftHand")
        [feature_train, label_train] = prepare_fisher_discriminant_analysis(bothFeet_train, leftHand_train, @MotorImagery_createBFvsLHfeature);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis(bothFeet_val, leftHand_val, @MotorImagery_createBFvsLHfeature);
        [model_BFvsLH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 4) = accuracy;
        disp("model_bothFeetVSrightHand")
        [feature_train, label_train] = prepare_fisher_discriminant_analysis(bothFeet_train, rightHand_train, @MotorImagery_createBFvsRHfeature);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis(bothFeet_val, rightHand_val, @MotorImagery_createBFvsRHfeature);
        [model_BFvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 5) = accuracy;

        disp("model_leftHandVSrightHand")
        [feature_train, label_train] = prepare_fisher_discriminant_analysis(leftHand_train, rightHand_train, @MotorImagery_createLHvsRHfeature);
        [feature_val, label_val] = prepare_fisher_discriminant_analysis(leftHand_val, rightHand_val, @MotorImagery_createLHvsRHfeature);
        [model_LHvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val);
        accuracies(fold, 6) = accuracy;
    end
    
    disp("OVO")
    accuracies(fold, 7) = MotorImagery_eval_OVO(leftHand_val, rightHand_val, bothHands_val, bothFeet_val, nbFilters,...
                                model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH);
    
end

disp("Result of 5-fold cross validation")
models = ["model_bothHandVSbothFeet", "model_bothHandVSleftHand", "model_bothHandVSrightHand",...
            "model_bothFeetVSleftHand", "model_bothFeetVSrightHand", "model_leftHandVSrightHand",...
            "OVO"];
if nbFilters ~= 0
    % CSP features
    for i=1:7
        disp(models(i))
        disp("CSP features - accuracy: mean " + mean(accuracies(1:nbFolds, i), 1) + " variance " + std(accuracies(1:nbFolds, i), 1))
    end
else
    % custom features
    for i=1:7
        disp(models(i))
        disp("Custom features - accuracy: mean " + mean(accuracies(1:nbFolds, i), 1) + " variance " + std(accuracies(1:nbFolds, i), 1))
    end
end

% train classifier on complete training data
training_fold = nbFolds + 1;
if nbFilters ~= 0
    % CSP features
    disp("model_bothHandVSbothFeet")
    cspMatrix_BHvsBF = calc_csp(data_bothHands, data_bothFeet);
    save('MotorImagery_cspMatrix_BHvsBF.mat', 'cspMatrix_BHvsBF');
    [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(data_bothHands, data_bothFeet, cspMatrix_BHvsBF, nbFilters);
    [model_BHvsBF, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 1) = accuracy;
    disp("model_bothHandVSleftHand")
    cspMatrix_BHvsLH = calc_csp(data_bothHands, data_leftHand);
    save('MotorImagery_cspMatrix_BHvsLH.mat', 'cspMatrix_BHvsLH');
    [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(data_bothHands, data_leftHand, cspMatrix_BHvsLH, nbFilters);
    [model_BHvsLH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 2) = accuracy;
    disp("model_bothHandVSrightHand")
    cspMatrix_BHvsRH = calc_csp(data_bothHands, data_rightHand);
    save('MotorImagery_cspMatrix_BHvsRH.mat', 'cspMatrix_BHvsRH');
    [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(data_bothHands, data_rightHand, cspMatrix_BHvsRH, nbFilters);
    [model_BHvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 3) = accuracy;

    disp("model_bothFeetVSleftHand")
    cspMatrix_BFvsLH = calc_csp(data_bothFeet, data_leftHand);
    save('MotorImagery_cspMatrix_BFvsLH.mat', 'cspMatrix_BFvsLH');
    [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(data_bothFeet, data_leftHand, cspMatrix_BFvsLH, nbFilters);
    [model_BFvsLH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 4) = accuracy;
    disp("model_bothFeetVSrightHand")
    cspMatrix_BFvsRH = calc_csp(data_bothFeet, data_rightHand);
    save('MotorImagery_cspMatrix_BFvsRH.mat', 'cspMatrix_BFvsRH');
    [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(data_bothFeet, data_rightHand, cspMatrix_BFvsRH, nbFilters);
    [model_BFvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 5) = accuracy;

    disp("model_leftHandVSrightHand")
    cspMatrix_LHvsRH = calc_csp(data_leftHand, data_rightHand);
    save('MotorImagery_cspMatrix_LHvsRH.mat', 'cspMatrix_LHvsRH');
    [feature_train, label_train] = prepare_fisher_discriminant_analysis_csp(data_leftHand, data_rightHand, cspMatrix_LHvsRH, nbFilters);
    [model_LHvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 6) = accuracy;
else
    % custom features
    disp("model_bothHandsVSbothFeed")
    [feature_train, label_train] = prepare_fisher_discriminant_analysis(data_bothHands, data_bothFeet, @MotorImagery_createBHvsBFfeature);
    [model_BHvsBF, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 1) = accuracy;
    disp("model_bothHandsVSleftHand")
    [feature_train, label_train] = prepare_fisher_discriminant_analysis(data_bothHands, data_leftHand, @MotorImagery_createBHvsLHfeature);
    [model_BHvsLH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 2) = accuracy;
    disp("model_bothHandsVSrightHand")
    [feature_train, label_train] = prepare_fisher_discriminant_analysis(data_bothHands, data_rightHand, @MotorImagery_createBHvsRHfeature);
    [model_BHvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 3) = accuracy;

    disp("model_bothFeetVSleftHand")
    [feature_train, label_train] = prepare_fisher_discriminant_analysis(data_bothFeet, data_leftHand, @MotorImagery_createBFvsLHfeature);
    [model_BFvsLH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 4) = accuracy;
    disp("model_bothFeetVSrightHand")
    [feature_train, label_train] = prepare_fisher_discriminant_analysis(data_bothFeet, data_rightHand, @MotorImagery_createBFvsRHfeature);
    [model_BFvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 5) = accuracy;

    disp("model_leftHandVSrightHand")
    [feature_train, label_train] = prepare_fisher_discriminant_analysis(data_leftHand, data_rightHand, @MotorImagery_createLHvsRHfeature);
    [model_LHvsRH, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_train, label_train);
    accuracies(training_fold, 6) = accuracy;
end
disp("OVO")
accuracies(training_fold, 7) = MotorImagery_eval_OVO(data_leftHand, data_rightHand, data_bothHands, data_bothFeet, nbFilters,...
                                model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH);


disp("Result of training on the complete training data")
models = ["model_bothHandVSbothFeet", "model_bothHandVSleftHand", "model_bothHandVSrightHand",...
            "model_bothFeetVSleftHand", "model_bothFeetVSrightHand", "model_leftHandVSrightHand",...
            "OVO"];
if nbFilters ~= 0
    % CSP features
    for i=1:7
        disp(models(i))
        disp("CSP features - accuracy on training data: " + accuracies(training_fold, i))
    end
else
    % custom features
    for i=1:7
        disp(models(i))
        disp("Custom features - accuracy on training data: " + accuracies(training_fold, i))
    end
end

end