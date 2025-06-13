clc; clear;

%% === Load Pre-Extracted YAMNet Features ===
fprintf("📂 Loading YAMNet features...\n");

load('src/feature_extraction/features_yamnet.mat', ...
    'trainFeatures', 'valFeatures', 'testFeatures');

% === Prepare sequences ===
fprintf("🧪 Preparing sequences...\n");

toSeq = @(features) deal( ...
    {features.embedding}', ...
    categorical({features.label})');

[XTrain, YTrain] = toSeq(trainFeatures);
[XVal,   YVal]   = toSeq(valFeatures);
[XTest,  YTest]  = toSeq(testFeatures);

%% === Define LSTM Network ===
inputSize = 1024;
numHiddenUnits = 128;
numClasses = numel(categories(YTrain));

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XVal, YVal}, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots','training-progress');

%% === Train LSTM ===
fprintf("🧠 Training LSTM model...\n");
lstmModel = trainNetwork(XTrain, YTrain, layers, options);

%% === Save Model ===
save('models/lstm_model_yamnet.mat', 'lstmModel');
fprintf("✅ LSTM model saved to models/lstm_model_yamnet.mat\n");

%% === Evaluate ===
fprintf("📈 Evaluating on test set...\n");
YPred = classify(lstmModel, XTest, 'MiniBatchSize', 32);

acc = sum(YPred == YTest) / numel(YTest);
fprintf("🎯 Test Accuracy: %.2f%%\n", acc * 100);

confusionchart(YTest, YPred);
title('LSTM Confusion Matrix (YAMNet Embeddings)');
