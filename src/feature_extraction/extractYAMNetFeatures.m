function extractYAMNetFeatures()
    % Load YAMNet model
    yamnet = load('src/yamnet/yamnet.mat');  % path to pretrained YAMNet
    yamnet = yamnet.yamnet;

    % Load preprocessed audio data
    load('src/preprocessing/processed_data.mat', ...
        'trainData', 'valData', 'testData');

    fprintf("🎧 Extracting YAMNet embeddings...\n");

    % Extract features
    trainFeatures = extractFromTable(trainData, yamnet);
    valFeatures   = extractFromTable(valData, yamnet);
    testFeatures  = extractFromTable(testData, yamnet);

    % Save extracted features
    save('src/feature_extraction/features_yamnet.mat', ...
        'trainFeatures', 'valFeatures', 'testFeatures');

    fprintf("✅ Saved YAMNet features to features_yamnet.mat\n");
end

% --- Helper: Extract YAMNet Embeddings from Table ---
function features = extractFromTable(dataTable, yamnet)
    numSamples = height(dataTable);
    features(numSamples) = struct('embedding', [], 'label', []);

    for i = 1:numSamples
        audio = dataTable.Audio{i};
        fs    = dataTable.SampleRate(i);
        label = dataTable.Label{i};

        % Ensure audio is mono and 16 kHz (YAMNet default)
        audio = resample(audio, 16000, fs);
        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end

        % Predict using YAMNet (returns scores, embeddings, etc.)
        [~, embeddings, ~] = yamnetPredict(yamnet, audio);

        features(i).embedding = embeddings;  % [T x 1024]
        features(i).label = label;
    end
end
