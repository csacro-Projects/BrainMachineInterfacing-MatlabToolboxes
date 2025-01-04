function [feature] = MotorImagery_createBFvsRHfeature(data)

[~, data] = calc_channel_timeseries(data, 256, 512, 4);

feature = [
     squeeze(min(data(29:82,13,:))),...
     squeeze(min(data(33:95,20,:)))
    ];

end

