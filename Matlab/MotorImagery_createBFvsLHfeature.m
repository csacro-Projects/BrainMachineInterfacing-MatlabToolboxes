function [feature] = MotorImagery_createBFvsLHfeature(data)

[~, data] = calc_channel_timeseries(data, 256, 512, 4);

feature = [
     squeeze(min(data(118:end,25,:))),...
     squeeze(min(data(44:85,24,:)))
    ];

end

