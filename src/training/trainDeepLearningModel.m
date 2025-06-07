function trainCNN(useMFCC)
%% 🤖 CNN Training Script with MFCC/Spectrogram switch
% Input: useMFCC = true for MFCC, false for Spectrogram
clc;

fprintf("📦 Loading features...\n");

load('src/feature_extraction/features_mfcc_spec.mat', 'trainFeatures', 'testFeatures');

%% 🔧 Prepare features
resizeTo = [64 64];  % Resize for CNN
fprintf("🔧 Preparing %s features...\n", ternary(useMFCC, 'MFCC', 'Spectrogram'));

% Process training features
trainData = arrayfun(@(x) struct( ...
    'features', imresize(useMFCC * x.mfcc + (~useMFCC) * x.spectrogram, resizeTo), ...
    'label', x.label), trainFeatures);

% Process test features
testData = arrayfun(@(x) struct( ...
    'features', imresize(useMFCC * x.mfcc + (~useMFCC) * x.spectrogram, resizeTo), ...
    'label', x.label), testFeatures);

XTrain = cat(4, trainData.features); % 64x64x1xN
YTrain = categorical({trainData.label});

XTest  = cat(4, testData.features);
YTest  = categorical({testData.label});

% Normalize
XTrain = rescale(XTrain);
XTest  = rescale(XTest);

%% 🧠 CNN Layers
inputSize = [resizeTo 1];
numClasses = numel(unique(YTrain));

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

%% ⚙️ Training Options
options = trainingOptions('adam', ...
    'MaxEpochs',15, ...
    'MiniBatchSize',32, ...
    'ValidationData',{XTest,YTest}, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress');

%% 🚀 Train Model
fprintf("🚀 Training CNN on %s features...\n", ternary(useMFCC, 'MFCC', 'Spectrogram'));
net = trainNetwork(XTrain, YTrain, layers, options);

% 🎯 Accuracy
YPred = classify(net, XTest);
acc = mean(YPred == YTest);
fprintf("✅ Accuracy: %.2f%%\n", acc*100);

figure;
confusionchart(YTest, YPred);
title(sprintf("CNN - %s Feature", ternary(useMFCC, 'MFCC', 'Spectrogram')));

%% 💾 Save model
modelName = ternary(useMFCC, 'cnn_MFCC_model.mat', 'cnn_Spectrogram_model.mat');
save(fullfile('models', modelName), 'net');
fprintf("💾 Model saved to: models/%s\n", modelName);

end

%% Helper: ternary-like operator
function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
