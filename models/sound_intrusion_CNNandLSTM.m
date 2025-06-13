%% === CONFIGURE PATHS ===
metaFile = 'esc50.csv';  % Must be in MATLAB current folder
audioFolder = 'audio_subset';  % Must be uploaded and unzipped

%% === LOAD METADATA ===
metadata = readtable(metaFile);
disp('✅ Metadata loaded successfully.');

%% === SELECT TARGET CLASSES ===
targetClasses = {'dog', 'engine', 'gun_shot', 'siren'};
ambientClasses = setdiff(unique(metadata.category), targetClasses);
metadata.category(ismember(metadata.category, ambientClasses)) = {'ambient'};
targetClasses = [targetClasses, 'ambient']; % Add ambient to target

% Filter only those classes
filteredData = metadata(ismember(metadata.category, targetClasses), :);
disp(['🎯 Selected ', num2str(height(filteredData)), ' audio files from target classes.']);

%% === FEATURE EXTRACTION ===
features = [];
labels = {};

disp('🔍 Extracting features...');

for i = 1:height(filteredData)
    filepath = fullfile(audioFolder, filteredData.filename{i});
    if ~isfile(filepath)
        warning('Missing file: %s. Skipping...', filepath);
        continue;
    end

    [audioIn, fs] = audioread(filepath);

    % Convert to mono
    if size(audioIn, 2) > 1
        audioIn = mean(audioIn, 2);
    end

    % Pad/trim to 5 sec
    targetLength = 5 * fs;
    audioIn = padarray(audioIn(1:min(end, targetLength)), targetLength - min(end, targetLength), 0, 'post');

    % Create mel spectrogram
    melSpec = melSpectrogram(audioIn, fs, 'WindowLength', 1024, 'OverlapLength', 512, 'NumBands', 64);
    features(:, :, 1, i) = log10(melSpec + eps);  % [bands x frames x 1 x samples]
    labels{i} = filteredData.category{i};
end

labels = categorical(labels);

%% === SPLIT INTO TRAIN / TEST ===
cv = cvpartition(labels, 'HoldOut', 0.3);
XTrain = features(:, :, 1, training(cv));
YTrain = labels(training(cv));
XTest = features(:, :, 1, test(cv));
YTest = labels(test(cv));
disp('📊 Data split into training and testing.');

%% === DEFINE CNN + LSTM ARCHITECTURE ===
numClasses = numel(categories(labels));

layers = [
    imageInputLayer([size(features,1) size(features,2) 1], 'Name', 'input')

    convolution2dLayer([3 3], 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([2 2], 'Stride', 2)

    convolution2dLayer([3 3], 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([2 2], 'Stride', 2)

    flattenLayer
    lstmLayer(64, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

%% === TRAIN MODEL ===
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 16, ...
    'ValidationData', {XTest, YTest}, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

disp('🧠 Training CNN+LSTM model...');
net = trainNetwork(XTrain, YTrain, layers, options);

%% === EVALUATE MODEL ===
YPred = classify(net, XTest);
acc = sum(YPred == YTest) / numel(YTest);
disp(['✅ Test Accuracy: ', num2str(acc * 100, '%.2f'), '%']);

figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');

%% === PREDICT ON NEW CLIP ===
newClip = 'test_clip.wav';  % Must be uploaded
[audioIn, fs] = audioread(newClip);

if size(audioIn, 2) > 1
    audioIn = mean(audioIn, 2);
end

targetLength = 5 * fs;
audioIn = padarray(audioIn(1:min(end, targetLength)), targetLength - min(end, targetLength), 0, 'post');

melSpec = melSpectrogram(audioIn, fs, 'WindowLength', 1024, 'OverlapLength', 512, 'NumBands', 64);
melLog = log10(melSpec + eps);

[YPredNew, scores] = classify(net, melLog);

%% === THRESHOLD LOGIC ===
threshold = 0.8;
[maxScore, maxIdx] = max(scores);
predictedLabel = string(YPredNew);

if maxScore < threshold
    predictedLabel = "ambient";
end

disp(['🔊 New Clip Prediction: ', predictedLabel, ' (Confidence: ', num2str(maxScore*100, '%.2f'), '%)']);

% Show top 5 class scores
[sortedScores, sortedIdx] = sort(scores, 'descend');
top5Labels = net.Layers(end).Classes(sortedIdx(1:5));
disp('📊 Top 5 Predictions:');
for i = 1:5
    disp([char(top5Labels(i)), ': ', num2str(sortedScores(i)*100, '%.2f'), '%']);
end
