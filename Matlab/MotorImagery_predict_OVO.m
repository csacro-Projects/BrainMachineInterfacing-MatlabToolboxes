function [result] = MotorImagery_predict_OVO(data, nbFilters, model_BHvsBF, model_BHvsLH, model_BHvsRH, model_BFvsLH, model_BFvsRH, model_LHvsRH)

    if nbFilters ~= 0
        % CSP matrices
        load('MotorImagery_cspMatrix_BHvsBF.mat', 'cspMatrix_BHvsBF');
        load('MotorImagery_cspMatrix_BHvsLH.mat', 'cspMatrix_BHvsLH');
        load('MotorImagery_cspMatrix_BHvsRH.mat', 'cspMatrix_BHvsRH');
        load('MotorImagery_cspMatrix_BFvsLH.mat', 'cspMatrix_BFvsLH');
        load('MotorImagery_cspMatrix_BFvsRH.mat', 'cspMatrix_BFvsRH');
        load('MotorImagery_cspMatrix_LHvsRH.mat', 'cspMatrix_LHvsRH');
        
        % CSP features
        model_BHvsBF_feature = apply_csp(data, cspMatrix_BHvsBF, nbFilters);
        model_BHvsLH_feature = apply_csp(data, cspMatrix_BHvsLH, nbFilters);
        model_BHvsRH_feature = apply_csp(data, cspMatrix_BHvsRH, nbFilters);
        model_BFvsLH_feature = apply_csp(data, cspMatrix_BFvsLH, nbFilters);
        model_BFvsRH_feature = apply_csp(data, cspMatrix_BFvsRH, nbFilters);
        model_LHvsRH_feature = apply_csp(data, cspMatrix_LHvsRH, nbFilters);
    else
        % custom features
        model_BHvsBF_feature = MotorImagery_createBHvsBFfeature(data);
        model_BHvsLH_feature = MotorImagery_createBHvsLHfeature(data);
        model_BHvsRH_feature = MotorImagery_createBHvsRHfeature(data);
        model_BFvsLH_feature = MotorImagery_createBFvsLHfeature(data);
        model_BFvsRH_feature = MotorImagery_createBFvsRHfeature(data);
        model_LHvsRH_feature = MotorImagery_createLHvsRHfeature(data);
    end
    
    % labels
    LEFT_HAND = 1;
    RIGHT_HAND = 2;
    BOTH_HANDS = 3;
    BOTH_FEET = 4;
    
    [model_BHvsBF_result, ~] = predict_fisher_discriminant_analysis(model_BHvsBF, model_BHvsBF_feature);
    model_BHvsBF_result(model_BHvsBF_result==-1) = BOTH_HANDS;
    model_BHvsBF_result(model_BHvsBF_result==1) = BOTH_FEET;
    
    [model_BHvsLH_result, ~] = predict_fisher_discriminant_analysis(model_BHvsLH, model_BHvsLH_feature);
    model_BHvsLH_result(model_BHvsLH_result==-1) = BOTH_HANDS;
    model_BHvsLH_result(model_BHvsLH_result==1) = LEFT_HAND;
    
    [model_BHvsRH_result, ~] = predict_fisher_discriminant_analysis(model_BHvsRH, model_BHvsRH_feature);
    model_BHvsRH_result(model_BHvsRH_result==-1) = BOTH_HANDS;
    model_BHvsRH_result(model_BHvsRH_result==1) = RIGHT_HAND;
    
    [model_BFvsLH_result, ~] = predict_fisher_discriminant_analysis(model_BFvsLH, model_BFvsLH_feature);
    model_BFvsLH_result(model_BFvsLH_result==-1) = BOTH_FEET;
    model_BFvsLH_result(model_BFvsLH_result==1) = LEFT_HAND;
    
    [model_BFvsRH_result, ~] = predict_fisher_discriminant_analysis(model_BFvsRH, model_BFvsRH_feature);
    model_BFvsRH_result(model_BFvsRH_result==-1) = BOTH_FEET;
    model_BFvsRH_result(model_BFvsRH_result==1) = RIGHT_HAND;
        
    [model_LHvsRH_result, ~] = predict_fisher_discriminant_analysis(model_LHvsRH, model_LHvsRH_feature);
    model_LHvsRH_result(model_LHvsRH_result==-1) = LEFT_HAND;
    model_LHvsRH_result(model_LHvsRH_result==1) = RIGHT_HAND;
    
    model_result = [model_BHvsBF_result, model_BHvsLH_result, model_BHvsRH_result,...
                        model_BFvsLH_result, model_BFvsRH_result, model_LHvsRH_result]';
    labels = [LEFT_HAND, RIGHT_HAND, BOTH_HANDS, BOTH_FEET]; % position matches label number
    counts = histc(model_result, labels);
    
    values = max(counts);
    result = zeros(size(values))';
    for i=1:length(values)
        idx = find(counts(:, i)==values(i));
        if length(idx) == 1
            result(i) = idx;
        elseif length(idx) == 2
            % in this case classifier comparing the options knows best
            disp("There are 2 options to select from for input at position " + i)
            if idx(1) == LEFT_HAND
                if idx(2) == RIGHT_HAND
                    result(i) = model_LHvsRH_result(i);
                elseif idx(2) == BOTH_HANDS
                    result(i) = model_BHvsLH_result(i);
                elseif idx(2) == BOTH_FEET
                    result(i) = model_BFvsLH_result(i);
                end
            elseif idx(1) == RIGHT_HAND
                if idx(2) == BOTH_HANDS
                    result(i) = model_BHvsRH_result(i);
                elseif idx(2) == BOTH_FEET
                    result(i) = model_BFvsRH_result(i);
                end
            elseif idx(1) == BOTH_HANDS
                result(i) = model_BHvsBF_result(i);
            end
        else % lenght(idx) == 3
            % could be improved
            disp("There are 3 options to select from for input at position " + i)
            result(i) = randsample(idx,1);
        end
    end

end