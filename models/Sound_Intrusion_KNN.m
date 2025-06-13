%% === CONFIG ===
metaFile = 'esc50.csv';     % CSV metadata file
audioFolder = 'audio_subset';  % Folder with audio files
threshold = 0.8;            % Confidence threshold

%% === Load Metadata ===
metadata = readtable(metaFile);
disp("✅ Metadata loaded.");

%% === Select & Relabel Target Classes ===
targetClasses = {'dog', 'engine', 'gun_shot', 'siren'};
ambientClasses = setdiff(unique(metadata.category), targetClasses);
metadata.category(ismember(metadata.category, ambientClasses)) = {'ambient'};
targetClasses = [targetClasses, 'ambient'];

filteredData = metadata(ismember(metadata.category, targetClasses), :);
disp(['🎯 Selected ', num2str(height(filteredData)), ' files.']);

%% === Load YAMNet Pretrained Model ===
yamnet = yamnetPretrained;
disp("🎵 YAMNet model loaded.");

%% === Feature Extraction using YAMNet ===
features = [];
labels = {};

for i = 1:height(filteredData)
    filePath = fullfile(audioFolder, filteredData.filename{i});
    if ~isfile(filePath)
        warning('Skipping missing file: %s', filePath);
        continue;
    end

    [audioIn, fs] = audioread(filePath);
    if size(audioIn,2) > 1
        audioIn = mean(audioIn,2);
    end

    % Resample if needed
    if fs ~= yamnet.SampleRate
        audioIn = resample(audioIn, yamnet.SampleRate, fs);
    end

    % Extract embeddings using YAMNet
    [~,~,scores,embeddings] = yamnet.predict(audioIn);
    meanEmbed = mean(embeddings,1);  % Average over time
    features = [features; meanEmbed];
    labels{end+1,1} = filteredData.category{i};
end

labels = categorical(labels);

%% === Train/Test Split ===
cv = cvpartition(labels, 'HoldOut', 0.3);
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest = features(test(cv), :);
YTest = labels(test(cv), :);
disp("🧪 Data split complete.");

%% === Train KNN Classifier ===
knnModel = fitcknn(XTrain, YTrain, 'NumNeighbors', 5);
disp("🧠 KNN model trained.");

%% === Evaluate on Test Data ===
YPred = predict(knnModel, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['✅ Test Accuracy: ', num2str(accuracy*100, '%.2f'), '%']);

figure;
confusionchart(YTest, YPred);
title('Confusion Matrix (KNN)');

%% === Predict on New Test Clip ===
testClip = 'test_clip.wav';
[audioIn, fs] = audioread(testClip);

if size(audioIn,2) > 1
    audioIn = mean(audioIn,2);
end

if fs ~= yamnet.SampleRate
    audioIn = resample(audioIn, yamnet.SampleRate, fs);
end

[~,~,~,embedding] = yamnet.predict(audioIn);
meanEmbedding = mean(embedding, 1);

[labelPred, scoreVec, ~, posterior] = predict(knnModel, meanEmbedding);

% Get class names
classNames = categories(labels);
[sortedScores, sortedIdx] = sort(posterior, 'descend');

% Threshold logic
[maxScore, maxIdx] = max(posterior);
if maxScore < threshold
    predictedLabel = "ambient";
else
    predictedLabel = classNames{maxIdx};
end

disp(['🔊 New Clip Prediction: ', predictedLabel, ' (', num2str(maxScore*100, '%.2f'), '% confidence)']);

% Display Top 5 Predictions
disp("📊 Top 5 Predictions:");
for i = 1:min(5, numel(classNames))
    fprintf("%s: %.2f%%\n", classNames{sortedIdx(i)}, sortedScores(i)*100);
end
