function [feature] = P300Speller_createFeature(data)

feature = [
    squeeze(mean(mean(data(53:62,6,:,:), 4))),...
    squeeze(mean(mean(data(54:62,8,:,:), 4)))
    ];

end

