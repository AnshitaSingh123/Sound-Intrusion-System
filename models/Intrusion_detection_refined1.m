%% === 1. LOAD METADATA AND AUDIO PATHS ===

metaFile = 'esc50.csv';  % Metadata CSV file placed in MATLAB path
audioFolder = 'audio_subset';  % Folder with audio files (~5-sec WAV clips)

% Read metadata table
metadata = readtable(metaFile);
disp('✅ Metadata loaded successfully.');

%% === 2. ADD AMBIENT CATEGORY FOR NON-TARGET CLASSES ===

targetClasses = {'gun_shot', 'siren', 'engine', 'dog'};
metadata.category(~ismember(metadata.category, targetClasses)) = {'ambient'};
filteredData = metadata;  % Now includes all rows with 5 total classes

disp(['🎯 Using ', num2str(height(filteredData)), ' audio samples from 5 categories.']);

%% === 3. EXTRACT MFCC FEATURES FROM AUDIO ===

features = [];
labels = {};

disp('🔍 Extracting MFCC features...');

for i = 1:height(filteredData)
    filename = fullfile(audioFolder, filteredData.filename{i});
    
    if ~isfile(filename)
        warning('File not found: %s', filename);
        continue;
    end

    [audioIn, fs] = audioread(filename);
    
    % Convert stereo to mono if needed
    if size(audioIn, 2) > 1
        audioIn = mean(audioIn, 2);
    end
    
    % Pad or trim to exactly 5 seconds
    targetLength = 5 * fs;
    if length(audioIn) < targetLength
        audioIn(end+1:targetLength) = 0;
    else
        audioIn = audioIn(1:targetLength);
    end
    
    % Compute MFCC and take mean across time
    coeffs = mfcc(audioIn, fs, 'NumCoeffs', 13);
    meanMFCC = mean(coeffs, 1);

    % Store result
    features = [features; meanMFCC];
    labels{end+1, 1} = filteredData.category{i};
end

labels = categorical(labels);

%% === 4. SPLIT INTO TRAIN AND TEST SET ===

cv = cvpartition(labels, 'HoldOut', 0.3);
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest = features(test(cv), :);
YTest = labels(test(cv));

%% === 5. TRAIN CLASSIFIER (SVM) ===

disp('🧠 Training SVM model...');
model = fitcecoc(XTrain, YTrain);
disp('✅ Model training completed.');

%% === 6. PREDICT AND EVALUATE ===

[YPred, scores] = predict(model, XTest);

% Accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['📊 Test Accuracy: ', num2str(accuracy * 100, '%.2f'), '%']);

% Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix - ESC-50 SVM Classifier');

% Show top-5 predictions for first test sample
[sortedScores, idx] = sort(scores(1,:), 'descend');
sortedLabels = model.ClassNames(idx);
disp('🔎 Top predictions for 1st test sample:');
for j = 1:min(5, numel(sortedLabels))
    fprintf('%s: %.2f%%\n', string(sortedLabels(j)), sortedScores(j)*100);
end

%% === 7. PREDICT A NEW CUSTOM AUDIO FILE ===

customFile = 'test_clip.wav';  % Place your own 5-sec WAV file here

if isfile(customFile)
    [newAudio, fs] = audioread(customFile);

    % Convert to mono if stereo
    if size(newAudio, 2) > 1
        newAudio = mean(newAudio, 2);
    end

    % Pad or trim
    targetLength = 5 * fs;
    if length(newAudio) < targetLength
        newAudio(end+1:targetLength) = 0;
    else
        newAudio = newAudio(1:targetLength);
    end

    % Feature extraction
    coeffs = mfcc(newAudio, fs, 'NumCoeffs', 13);
    featVec = mean(coeffs, 1);

    % Predict
    [predictedLabel, scoreVec] = predict(model, featVec);
    [maxScore, idx] = max(scoreVec);
    
    % Thresholding logic
    threshold = 0.80;
    if maxScore < threshold
        predictedLabel = 'ambient';
        disp(['⚠️ Low confidence (', num2str(maxScore*100, '%.2f'), '%). Classified as: ambient']);
    else
        disp(['✅ Predicted class: ', string(predictedLabel), ' (', num2str(maxScore*100, '%.2f'), '%)']);
    end

    % Show all class scores
    disp('🔬 Class confidence scores:');
    for j = 1:numel(model.ClassNames)
        fprintf('%s: %.2f%%\n', string(model.ClassNames(j)), scoreVec(j)*100);
    end
else
    disp('⚠️ No custom test clip found: test_clip.wav');
end
