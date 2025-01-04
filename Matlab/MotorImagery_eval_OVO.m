function [accuracy] = MotorImagery_eval_OVO(data_LH, data_RH, data_BH, data_BF, nbFilters,...
                            model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH)

LEFT_HAND = 1;
RIGHT_HAND = 2;
BOTH_HANDS = 3;
BOTH_FEET = 4;
nb = size(data_LH, 3) + size(data_RH, 3) + size(data_BH, 3) + size(data_BF, 3);

result = MotorImagery_predict_OVO(data_LH, nbFilters, model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH);
nb_correct = length(find(result == LEFT_HAND));
result = MotorImagery_predict_OVO(data_RH, nbFilters, model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH);
nb_correct = nb_correct + length(find(result == RIGHT_HAND));
result = MotorImagery_predict_OVO(data_BH, nbFilters, model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH);
nb_correct = nb_correct + length(find(result == BOTH_HANDS));
result = MotorImagery_predict_OVO(data_BF, nbFilters, model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH);
nb_correct = nb_correct + length(find(result == BOTH_FEET));

accuracy = nb_correct / nb;
disp("accuracy of OVO multi-class model is " + accuracy)

end