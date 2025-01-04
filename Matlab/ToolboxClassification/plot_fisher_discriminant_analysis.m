function [] = plot_fisher_discriminant_analysis(model, feature, label)
%plot_fisher_discriminant_analysis
%plots the data and the decision boundary
%   model: model created by train_fisher_discriminant_analysis
%   feature: feature vector for the evaluation with size(feature) = [nbMatches, nbFeatures]
%   label: ground truth label for the features

figure('Name', 'Classification with Fischer discriminant anaysis (first two features)')
hold on;
% feature points
plot(feature(label==-1,1), feature(label==-1,2), 'o', 'color', [1 1 1], 'MarkerFaceColor', 'r', 'markerSize', 8)
plot(feature(label==1,1), feature(label==1,2), 'o', 'color', [1 1 1], 'MarkerFaceColor', 'b', 'markerSize', 8)
% decision boundary
K = model.Coeffs(1,2).Const; % bias
L = model.Coeffs(1,2).Linear; % coefficients
f =@(x1,x2) K + L(1)*x1 + L(2)*x2;
decision_boundary = fimplicit(f,[min(min(feature(:,1))) max(max(feature(:,1))) min(min(feature(:,2))) max(max(feature(:,2)))]);
decision_boundary.Color = [0 0 0];
decision_boundary.LineWidth = 2;
hold off;

end

