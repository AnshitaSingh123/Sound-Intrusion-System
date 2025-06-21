clc; clear;

%% === Load Features ===
fprintf("🔁 Loading pre-extracted MFCC features...\n");
featurePath = fullfile('src', 'feature_extraction', 'features_mfcc_spec.mat');
if ~isfile(featurePath)
    error("❌ Features file not found at: %s", featurePath);
end
load(featurePath, 'trainFeatures', 'testFeatures');
fprintf("✅ Features loaded.\n");

%% === Clean + Pad Training Set ===
fprintf("📦 Preparing padded training set...\n");
XTrain = {};
YTrain = {};
for i = 1:numel(trainFeatures)
    x = trainFeatures(i).mfcc;
    if isempty(x) || ~isnumeric(x), continue; end
    x = padToLength(enforceTx13(x), 200);  % [200 x 13]
    x = x';  % Transpose to [13 x 200]
    XTrain{end+1,1} = x;
    YTrain{end+1,1} = char(trainFeatures(i).label);
end
YTrain = categorical(YTrain);
fprintf("✅ %d valid training samples\n", numel(XTrain));

%% === Clean + Pad Test Set ===
fprintf("📦 Preparing padded test set...\n");
XTest = {};
YTest = {};
for i = 1:numel(testFeatures)
    x = testFeatures(i).mfcc;
    if isempty(x) || ~isnumeric(x), continue; end
    x = padToLength(enforceTx13(x), 200);  % [200 x 13]
    x = x';  % Transpose to [13 x 200]
    XTest{end+1,1} = x;
    YTest{end+1,1} = char(testFeatures(i).label);
end
YTest = categorical(YTest);
fprintf("✅ %d valid test samples\n", numel(XTest));
fprintf("🧩 All sequences padded to [13 x 200]\n");

%% === Final Sanity Check ===
if isempty(XTrain) || isempty(YTrain)
    error("❌ No training data after preprocessing. Check MFCCs or padding logic.");
end

XTrain = cellfun(@(x) cast(x, 'single'), XTrain, 'UniformOutput', false);
XTest  = cellfun(@(x) cast(x, 'single'), XTest,  'UniformOutput', false);

XTrain = reshape(XTrain, [], 1);  % Enforce column cell
XTest = reshape(XTest, [], 1);

fprintf("👀 Sample shape check: [%d x %d]\n", size(XTrain{1},1), size(XTrain{1},2));

%% === Define LSTM ===
inputSize = 13;
numHiddenUnits = 64;
numClasses = numel(categories(YTrain));

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'SequenceLength', 'longest', ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% === Train Model ===
fprintf("🚀 Training LSTM model...\n");
lstmModel = trainNetwork(XTrain, YTrain, layers, options);

%% === Save Model ===
if ~exist('models', 'dir')
    mkdir('models');
end
save(fullfile('models', 'lstm_model.mat'), 'lstmModel');
fprintf("💾 Saved to models/lstm_model.mat\n");

%% === Evaluate ===
fprintf("📊 Evaluating...\n");
YPred = classify(lstmModel, XTest);
acc = mean(YPred == YTest);
fprintf("✅ Test Accuracy: %.2f%%\n", acc * 100);

%% === Confusion Matrix ===
figure;
confusionchart(YTest, YPred);
title("LSTM Confusion Matrix (MFCC Padded)");

%% === Helper: Force [T x 13] ===
function x = enforceTx13(x)
    if size(x,1) == 13 && size(x,2) ~= 13
        x = x';  % transpose if [13 x T] → [T x 13]
    end
end

%% === Helper: Pad or Truncate to targetT ===
function x = padToLength(x, targetT)
    [T, D] = size(x);
    if T == targetT
        return;
    elseif T > targetT
        x = x(1:targetT, :);  % truncate
    else
        x = [x; zeros(targetT - T, D)];  % pad
    end
end