file_list = dir('ml_data\*\audio\*.wav');

for n = 1:length(file_list)
    disp(n);
    filepath = strcat(file_list(n).folder, '\', file_list(n).name);
    [x, fs] = audioread(filepath);

    x = sum(x, 2);  % sum to mono if needed

    [~, ~, perc] = acousticLoudness(x, fs, 'TimeVarying', true);
    file_list(n).loundess_n5 = perc(2);
    file_list(n).roughness = mean(acousticRoughness(x, fs));
    file_list(n).sharpness = acousticSharpness(x, fs);
    file_list(n).fluctuation = mean(acousticFluctuation(x, fs));
end

writetable(struct2table(file_list), 'psychoacoustic_metrics.csv');