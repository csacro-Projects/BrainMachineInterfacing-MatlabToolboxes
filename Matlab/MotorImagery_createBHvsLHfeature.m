function [feature] = MotorImagery_createBHvsLHfeature(data)

[~, data] = calc_channel_timeseries(data, 256, 512, 4);

feature = [
     squeeze(min(data(30:end,13,:))),...
     squeeze(min(data(150:end,20,:)))
    ];

end

