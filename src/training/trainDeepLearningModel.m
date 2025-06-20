%% 🤖 UrbanSound8K CNN Training Script with MFCC/Spectrogram Switch
% Set useMFCC = true for MFCC, false for Spectrogram
useMFCC = true;  % 🔄 Change to false to use spectrogram
clc;
fprintf("📦 Loading UrbanSound8K features...\n");
load('src/feature_extraction/features_mfcc_spec.mat', 'trainFeatures', 'testFeatures');
%% 🔧 Resize and prepare dataset
resizeTo = [64 64];  % CNN expects fixed input size
fprintf("🔧 Preparing %s features...\n", ternary(useMFCC, 'MFCC', 'Spectrogram'));

% Process training features
trainData = arrayfun(@(x) struct( ...
    'features', imresize( ...
        ternary(useMFCC, x.mfcc, x.spec), ...
        resizeTo), ...
    'label', x.label), trainFeatures); % Correctly uses 'trainFeatures'

% Process test features
% This block must come AFTER 'testFeatures' is loaded, and BEFORE XTest/YTest are formed
testData = arrayfun(@(x) struct( ...
    'features', imresize( ...
        ternary(useMFCC, x.mfcc, x.spec), ...
        resizeTo), ...
    'label', x.label), testFeatures); % Correctly uses 'testFeatures'

% Build input/output arrays for CNN
XTrain = cat(4, trainData.features);  % [64x64x1xN]
YTrain = categorical({trainData.label});
XTest  = cat(4, testData.features);  % Now testData is defined!
YTest  = categorical({testData.label});
% Normalize input (0-1 range)
XTrain = rescale(XTrain);
XTest  = rescale(XTest);
%% 🧠 Define CNN Architecture
inputSize = [resizeTo 1];
numClasses = numel(categories(YTrain));
layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];
%% ⚙️ Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {XTest, YTest}, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');
%% 🚀 Train the Network
fprintf("🚀 Training CNN on %s features for UrbanSound8K...\n", ternary(useMFCC, 'MFCC', 'Spectrogram'));
net = trainNetwork(XTrain, YTrain, layers, options);
%% 🎯 Evaluate Accuracy
YPred = classify(net, XTest);
acc = mean(YPred == YTest);
fprintf("✅ Test Accuracy: %.2f%%\n", acc * 100);
figure;
confusionchart(YTest, YPred);
title(sprintf("CNN - UrbanSound8K (%s)", ternary(useMFCC, 'MFCC', 'Spectrogram')));
%% 💾 Save Trained Model
modelName = ternary(useMFCC, 'cnn_UrbanSound_MFCC.mat', 'cnn_UrbanSound_Spectrogram.mat');
save(fullfile('models', modelName), 'net');
fprintf("💾 Model saved as: models/%s\n", modelName);
%% Helper: ternary-like function
function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end