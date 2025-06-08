function extractFeatures()
    % === Load Preprocessed Audio ===
    load('src/preprocessing/processed_data.mat', 'trainData', 'valData', 'testData');

    fprintf("🔍 Extracting MFCCs and Spectrograms...\n");

    % === Extract Features ===
    trainFeatures = extractFromTable(trainData);
    valFeatures = extractFromTable(valData);
    testFeatures = extractFromTable(testData);

    % === Save Features ===
    save('src/feature_extraction/features_mfcc_spec.mat', ...
        'trainFeatures', 'valFeatures', 'testFeatures');

    fprintf("✅ Feature extraction complete. Saved to features_mfcc_spec.mat\n");
end

% --- Helper Function: Extract MFCC + Spectrogram from Data Table ---
function features = extractFromTable(dataTable)
    numSamples = height(dataTable);
    features = struct('mfcc', [], 'spec', [], 'label', []);

    for i = 1:numSamples
        audio = dataTable.Audio{i};
        fs = dataTable.SampleRate(i);
        label = dataTable.Label{i};

        % === MFCC Features ===
        coeffs = mfcc(audio, fs, 'LogEnergy','Ignore');  % [N x 13]

        % === Spectrogram ===
        [s, ~, ~] = melSpectrogram(audio, fs);  % [mel x frames]

        % Store
        features(i).mfcc = coeffs';
        features(i).spec = s;
        features(i).label = label;
    end
end
