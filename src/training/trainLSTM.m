% File: src/training/trainLSTM.m
% Purpose: Train and evaluate an LSTM using MFCC features

clc; clear;

%% === Load Pre-Extracted MFCC Features ===
fprintf("\Loading pre-extracted MFCC features...\n");
if isfile('src/feature_extraction/features_mfcc_spec.mat')
    fprintf("\Features file found. Proceeding to load...\n");
else
    error("\Features file not found. Please check the path.");
end
load('src/feature_extraction/features_mfcc_spec.mat', 'trainFeatures', 'testFeatures');

%% === Prepare Sequence Data ===
fprintf("\Preparing training set...\n");
XTrain = cellfun(@(x) x', {trainFeatures.mfcc}, 'UniformOutput', false);  % Each 13xT -> Tx13
YTrain = categorical({trainFeatures.label});

fprintf("\Preparing test set...\n");
XTest = cellfun(@(x) x', {testFeatures.mfcc}, 'UniformOutput', false);
YTest = categorical({testFeatures.label});

%% === Define LSTM Network ===
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
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest, YTest}, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% === Train LSTM Model ===
fprintf("\Training LSTM model...\n");
lstmModel = trainNetwork(XTrain, YTrain, layers, options);

%% === Save Trained Model ===
modelPath = fullfile('models', 'lstm_model.mat');
save(modelPath, 'lstmModel');
fprintf("\LSTM model saved at: %s\n", modelPath);

%% === Evaluate Model ===
fprintf("\Evaluating model on test set...\n");
YPred = classify(lstmModel, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf("\Test Accuracy: %.2f%%\n", accuracy * 100);

%% === Plot Confusion Matrix ===
figure;
confusionchart(YTest, YPred);
title('LSTM Confusion Matrix (MFCC Features)');
