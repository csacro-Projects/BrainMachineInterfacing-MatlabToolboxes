function [feature] = MotorImagery_createBHvsRHfeature(data)

[~, data] = calc_channel_timeseries(data, 256, 512, 4);

feature = [
     squeeze(min(data(47:end,24,:))),...
     squeeze(min(data(48:84,25,:)))
    ];

end

