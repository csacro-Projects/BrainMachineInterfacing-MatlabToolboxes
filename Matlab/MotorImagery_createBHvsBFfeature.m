function [feature] = MotorImagery_createBHvsBFfeature(data)

[~, data] = calc_channel_timeseries(data, 256, 512, 4);

feature = [
     squeeze(min(data(30:80,13,:))),...
     squeeze(min(data(33:86,20,:)))
    ];

end

