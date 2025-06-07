function features = extractFromTable(dataTable)
    numSamples = height(dataTable);
    features = struct('mfcc', [], 'spectrogram', [], 'label', []);

    for i = 1:numSamples
        audio = dataTable.Audio{i};
        fs = dataTable.SampleRate(i);
        label = dataTable.Label{i};

        % === MFCC Features ===
        coeffs = mfcc(audio, fs, 'LogEnergy','Ignore');  % [N x 13]

        % === Spectrogram ===
        [s, ~, ~] = melSpectrogram(audio, fs);  % [mel x frames]

        % Store both feature types with consistent field names
        features(i).mfcc = coeffs';               % [13 x T]
        features(i).spectrogram = s;              % [mel x frames]
        features(i).label = label;
    end
end
