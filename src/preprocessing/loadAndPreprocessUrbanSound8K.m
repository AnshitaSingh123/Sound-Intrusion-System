function loadAndPreprocessUrbanSound8K()
    % === Paths ===
    dataPath = fullfile('data', 'UrbanSound8K');
    audioPath = fullfile(dataPath, 'audio');
    metadataFile = fullfile(dataPath, 'metadata', 'UrbanSound8K.csv');

    % === Load Metadata ===
    metadata = readtable(metadataFile);

    % === Filter Classes of Interest ===
    selectedClasses = {'dog_bark', 'gun_shot', 'siren', 'engine_idling'};
    filteredData = metadata(ismember(metadata.class, selectedClasses), :);

    % === Initialize Storage ===
    audioData = {};
    labels = {};
    fsList = [];

    fprintf("Loading audio files...\n");

    % === Loop Through Filtered Files ===
    for i = 1:height(filteredData)
        try
            row = filteredData(i, :);
            fold = sprintf('fold%d', row.fold);
            filename = row.slice_file_name{1};
            label = row.class{1};

            filePath = fullfile(audioPath, fold, filename);
            [y, fs] = audioread(filePath);

            % Normalize audio signal [-1, 1]
            y = y / max(abs(y) + eps);

            % Resample to a fixed rate (optional for uniformity)
            targetFs = 22050;
            if fs ~= targetFs
                y = resample(y, targetFs, fs);
                fs = targetFs;
            end

            audioData{end+1} = y;
            labels{end+1} = label;
            fsList(end+1) = fs;
        catch ME
            warning("Error loading %s: %s", filename, ME.message);
        end
    end

    % === Convert to Table ===
    dataTable = table(audioData', labels', fsList', ...
        'VariableNames', {'Audio', 'Label', 'SampleRate'});

    % === Shuffle Data ===
    rng(1);  % for reproducibility
    dataTable = dataTable(randperm(height(dataTable)), :);

    % === Split into Train / Val / Test (70/15/15) ===
    numSamples = height(dataTable);
    idxTrain = 1:round(0.7 * numSamples);
    idxVal   = (round(0.7 * numSamples) + 1):round(0.85 * numSamples);
    idxTest  = (round(0.85 * numSamples) + 1):numSamples;

    trainData = dataTable(idxTrain, :);
    valData   = dataTable(idxVal, :);
    testData  = dataTable(idxTest, :);

    % === Save Processed Data ===
    save('src/preprocessing/processed_data.mat', ...
        'trainData', 'valData', 'testData');

    fprintf("✅ Preprocessing complete. Saved to processed_data.mat\n");
end  % <<< Make sure this 'end' exists and is not commented or broken
