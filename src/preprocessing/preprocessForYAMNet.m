function preprocessForYAMNet()
    % Preprocess UrbanSound8K dataset for YAMNet feature extraction
    fprintf('📂 Preprocessing audio for YAMNet...\n');

    % === Set Paths ===
    scriptDir = fileparts(mfilename('fullpath'));
    repoRoot = fullfile(scriptDir, '..', '..'); % Go up 2 levels
    dataPath = fullfile(repoRoot, 'data', 'UrbanSound8K');
    audioPath = fullfile(dataPath, 'audio');
    metadataFile = fullfile(dataPath, 'metadata', 'UrbanSound8K.csv');

    % === Validate Metadata Path ===
    if ~isfile(metadataFile)
        error('❌ Metadata file not found at: %s', metadataFile);
    end

    % === Load Metadata ===
    metadata = readtable(metadataFile);

    % === Filter Target Classes ===
    selectedClasses = {'dog_bark', 'gun_shot', 'siren', 'engine_idling'};
    filteredData = metadata(ismember(metadata.class, selectedClasses), :);

    % === Initialize Storage ===
    audioData = {};
    labels = {};
    fsList = [];

    fprintf('🎧 Loading and preprocessing audio files...\n');

    % === YAMNet-Specific Parameters ===
    targetFs = 16000; % YAMNet requires 16 kHz
    minLength = 160; % Minimum 10ms at 16 kHz

    for i = 1:height(filteredData)
        try
            row = filteredData(i, :);
            fold = sprintf('fold%d', row.fold);
            filename = string(row.slice_file_name);
            label = string(row.class);

            filePath = fullfile(audioPath, fold, filename);
            [y, fs] = audioread(filePath);

            % Convert to mono
            if size(y, 2) > 1
                y = mean(y, 2);
            end

            % Resample to 16 kHz
            y = resample(y, targetFs, fs);

            % Normalize to [-1, 1]
            maxAmp = max(abs(y));
            if maxAmp > 0
                y = y / maxAmp;
            else
                warning('Silent audio for %s: max amplitude=0. Using low-amplitude noise.', filename);
                y = randn(max(minLength, length(y)), 1) * 0.01; % Low-amplitude noise
            end

            % Ensure minimum length
            if length(y) < minLength
                y = [y; zeros(minLength - length(y), 1)];
            end

            % Verify audio
            if any(isnan(y(:))) || any(isinf(y(:)))
                warning('Invalid audio (NaN/Inf) for %s. Skipping.', filename);
                continue;
            end
            if max(abs(y)) < 0.01
                warning('Low amplitude audio for %s: max amplitude=%f.', filename, max(abs(y)));
            end

            audioData{end+1} = y;
            labels{end+1} = label;
            fsList(end+1) = targetFs;
        catch e
            warning('⚠️ Error processing %s: %s', filename, e.message);
        end
    end

    % === Convert to Table & Shuffle ===
    dataTable = table(audioData', labels', fsList', ...
        'VariableNames', {'Audio', 'Label', 'SampleRate'});
    rng(1); % For reproducibility
    dataTable = dataTable(randperm(height(dataTable)), :);

    % === Train/Val/Test Split ===
    n = height(dataTable);
    idxTrain = 1:round(0.7 * n);
    idxVal = (round(0.7 * n) + 1):round(0.85 * n);
    idxTest = (round(0.85 * n) + 1):n;

    trainData = dataTable(idxTrain, :);
    valData = dataTable(idxVal, :);
    testData = dataTable(idxTest, :);

    % === Verify Data ===
    fprintf('Train samples: %d\n', height(trainData));
    fprintf('Val samples: %d\n', height(valData));
    fprintf('Test samples: %d\n', height(testData));

    % === Save ===
    save(fullfile(scriptDir, 'processed_data_yamnet.mat'), ...
        'trainData', 'valData', 'testData');

    fprintf('✅ Preprocessing complete. Saved to processed_data_yamnet.mat\n');
end