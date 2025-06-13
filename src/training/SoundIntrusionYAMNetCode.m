% === SECTION 1: SETUP AND DATA LOADING ===
% Path to metadata and audio folder (upload via MATLAB web interface)
metaFile = 'esc50.csv';                     % Ensure this is uploaded in the same directory
audioFolder = 'audio_subset';              % Folder must contain target .wav clips
yamnetModel = 'yamnet.mat';                % Download from https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

% Load metadata
metadata = readtable(metaFile);
disp("📄 Metadata loaded successfully.");

% Load YAMNet model
load(yamnetModel, 'yamnet');
disp("🧠 YAMNet model loaded.");

% === SECTION 2: CLASS SELECTION & PREPARATION ===
targetClasses = {'dog', 'siren', 'engine', 'gun_shot'};  % Define classes of interest
ambientClass = 'ambient';                                % Additional category

% Filter metadata for target + ambient class setup
filteredData = metadata(ismember(metadata.category, targetClasses), :);
disp(['🎯 Selected ', num2str(height(filteredData)), ' audio files from target classes.']);

% === SECTION 3: FEATURE EXTRACTION USING YAMNET ===
features = [];
labels = {};

disp("🔍 Extracting embeddings using YAMNet...");

for i = 1:height(filteredData)
    audioPath = fullfile(audioFolder, filteredData.filename{i});

    if ~isfile(audioPath)
        warning("⚠️ Missing audio: %s", audioPath);
        continue;
    end

    % Read audio
    [audioIn, fs] = audioread(audioPath);
    if size(audioIn, 2) > 1
        audioIn = mean(audioIn, 2);  % Convert stereo to mono
    end

    % Resample to 16 kHz for YAMNet
    if fs ~= 16000
        audioIn = resample(audioIn, 16000, fs);
        fs = 16000;
    end

    % Get YAMNet embeddings (features)
    [scores, embeddings, ~] = yamnet.predict(audioIn);

    % Average embedding across time
    meanEmbedding = mean(embeddings, 1);

    % Store features and labels
    features = [features; meanEmbedding];
    labels{end+1, 1} = filteredData.category{i};
end

labels = categorical(labels);

% === SECTION 4: TRAIN / TEST SPLIT ===
cv = cvpartition(labels, 'HoldOut', 0.3);
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest = features(test(cv), :);
YTest = labels(test(cv));

% === SECTION 5: TRAIN CLASSIFIER ===
disp("📊 Training ECOC SVM on YAMNet embeddings...");
model = fitcecoc(XTrain, YTrain);

% === SECTION 6: TEST PREDICTION & AMBIENT THRESHOLDING ===
[YPred, score] = predict(model, XTest);
[maxScore, maxIdx] = max(score, [], 2);
threshold = 0.80;  % Confidence threshold

YPredFinal = YPred;
belowThreshold = maxScore < threshold;
YPredFinal(belowThreshold) = categorical({ambientClass});

% === SECTION 7: RESULTS ===
accuracy = sum(YPredFinal == YTest) / numel(YTest);
disp(['✅ Final Accuracy (with ambient): ', num2str(accuracy * 100, '%.2f'), '%']);

% === SECTION 8: CONFUSION MATRIX ===
figure;
confusionchart(YTest, YPredFinal);
title('Confusion Matrix with Ambient Detection');

% === SECTION 9: DISPLAY TOP-5 SCORES FOR A SAMPLE ===
sampleIdx = 1;
disp("🔍 Top-5 Prediction Scores for First Test Sample:");
[sortedScores, sortedIdx] = sort(score(sampleIdx, :), 'descend');
classList = categories(YTrain);
for k = 1:min(5, length(sortedIdx))
    fprintf(" %s: %.2f%%\n", classList{sortedIdx(k)}, sortedScores(k)*100);
end

% === SECTION 10: CUSTOM AUDIO CLIP PREDICTION ===
customAudioPath = 'new_test_clip.wav';  % Upload your 5-sec test file
[audioIn, fs] = audioread(customAudioPath);

if size(audioIn, 2) > 1
    audioIn = mean(audioIn, 2);  % Convert to mono
end

if fs ~= 16000
    audioIn = resample(audioIn, 16000, fs);
    fs = 16000;
end

% Use YAMNet to extract embedding
[~, embeddings, ~] = yamnet.predict(audioIn);
meanEmbedding = mean(embeddings, 1);

% Predict label + score
[labelPred, scorePred] = predict(model, meanEmbedding);
[maxVal, idx] = max(scorePred);
classList = categories(YTrain);

% Apply threshold for ambient detection
if maxVal < threshold
    finalLabel = 'ambient';
else
    finalLabel = classList{idx};
end

fprintf("🔉 Prediction for '%s': %s (Confidence: %.2f%%)\n", ...
    customAudioPath, finalLabel, maxVal * 100);
