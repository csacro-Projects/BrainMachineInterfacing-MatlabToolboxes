function [CSPmatrix] = calc_csp(data_class1, data_class2)
%calc_csp
%returns the features calculated with csp algorithm and the labels
%   data_class1: data for the first class with size(data) = [nbSamples, nbChannels, nbMatchesClass1]
%   data_class2: data for the first class with size(data) = [nbSamples, nbChannels, nbMatchesClass2]

if size(size(data_class1), 2) ~= 3
    error("data_class1 does not have 3 dimensions")
end
if size(size(data_class2), 2) ~= 3
    error("data_class2 does not have 3 dimensions")
end
[~, nbChannels1, nbMatchesClass1] = size(data_class1);
[~, nbChannels2, nbMatchesClass2] = size(data_class2);
if nbChannels1 ~= nbChannels2
    error("amount of channels must be the same for both classes")
end
nbChannels = nbChannels1;

% normalized spatial covariance matrices
covarianceClass1=zeros(nbChannels, nbChannels, nbMatchesClass1);
for m = 1:nbMatchesClass1
    data_data = data_class1(:, :, m)' * data_class1(: ,:, m);
    covarianceClass1(:, :, m) = (data_data)./trace(data_data);
end
average_covarianceClass1 = mean(covarianceClass1, 3);

covarianceClass2 = zeros(nbChannels, nbChannels, nbMatchesClass2);
for m = 1:nbMatchesClass2
    data_data = data_class2(:, :, m)' * data_class2(: ,:, m);
    covarianceClass2(:, :, m)=(data_data)./trace(data_data);
end
average_covarianceClass2 = mean(covarianceClass2, 3);

% CSP matrix (with eigenvalues sorted in desceding order)
[eigenVec, eigenVal] = eig(average_covarianceClass1, average_covarianceClass2);
[~, idx_descend] = sort(diag(eigenVal), 'descend');
CSPmatrix = eigenVec(:, idx_descend)';

end