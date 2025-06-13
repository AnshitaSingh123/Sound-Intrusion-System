% File: src/training/trainCNN.m
% Purpose: Train and evaluate a basic CNN using MFCC features

clc; clear;

%% === Load MFCC Features ===
fprintf("📂 Loading pre-extracted MFCC features...\n");

featureFile = 'src/feature_extraction/features_mfcc_spec.mat';
if ~isfile(featureFile)
    error("❌ Features file not found. Run feature extraction first.");
end
load(featureFile, 'trainFeatures', 'testFeatures');

%% === Prepare Data for CNN ===
% Convert MFCC (13 x T) to 13 x T x 1 (like grayscale image)
preprocess = @(features) cat(3, features.mfcc, zeros(size(features.mfcc,1), size(features.mfcc,2), 0));

prepareData = @(dataStruct) arrayfun(@(f) struct( ...
    'X', preprocess(f), ...
    'Y', categorical(f.label)), dataStruct);

trainSet = prepareData(trainFeatures);
testSet = prepareData(testFeatures);

% Convert to image datastore
XTrain = cat(4, trainSet.X);
YTrain = categorical({trainSet.Y});

XTest = cat(4, testSet.X);
YTest = categorical({testSet.Y});

fprintf("✅ Training samples: %d\n", size(XTrain, 4));
fprintf("✅ Test samples: %d\n", size(XTest, 4));

%% === Define CNN Architecture ===
inputSize = [13, size(XTrain, 2), 1];
numClasses = numel(unique(YTrain));

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer([3 3], 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([2 2], 'Stride', 2)

    convolution2dLayer([3 3], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    globalAveragePooling2dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationData', {XTest, YTest});

%% === Train CNN ===
fprintf("🧠 Training CNN...\n");
cnnModel = trainNetwork(XTrain, YTrain, layers, options);

%% === Save Model ===
save(fullfile('models', 'cnn_model_mfcc.mat'), 'cnnModel');
fprintf("✅ CNN model saved to models/cnn_model_mfcc.mat\n");

%% === Evaluate Model ===
YPred = classify(cnnModel, XTest);
acc = mean(YPred == YTest);
fprintf("🎯 Test Accuracy: %.2f%%\n", acc * 100);

% Plot confusion matrix
figure;
confusionchart(YTest, YPred);
title('CNN Confusion Matrix (MFCC Input)');
