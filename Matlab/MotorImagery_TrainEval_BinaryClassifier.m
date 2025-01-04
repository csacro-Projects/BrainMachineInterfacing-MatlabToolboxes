function [model_fisher, accuracy] = MotorImagery_TrainEval_BinaryClassifier(feature_train, label_train, feature_val, label_val)

    model_fisher = train_fisher_discriminant_analysis(feature_train, label_train);
    eval_fisher_discriminant_analysis(model_fisher, feature_train, label_train);
    
    [~, accuracy] = eval_fisher_discriminant_analysis(model_fisher, feature_val, label_val);

end