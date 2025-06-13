% File: src/training/trainKNN.m
% Purpose: Train and evaluate a KNN using MFCC features

clc; clear;

%% === Load Pre-Extracted MFCC Features ===
fprintf("📂 Loading pre-extracted MFCC features...\n");
if isfile('src/feature_extraction/features_mfcc_spec.mat')
    fprintf("✅ Features file found. Proceeding to load...\n");
else
    error("❌ Features file not found. Please check the path.");
end
load('src/feature_extraction/features_mfcc_spec.mat', ...
    'trainFeatures', 'testFeatures');

%% === Convert struct array to table with averaged MFCCs ===
mfccToTable = @(featureStruct) ...
    struct2table(arrayfun(@(x) struct( ...
        'Features', mean(x.mfcc, 2)', ...
        'Label', x.label), ...
        featureStruct));

fprintf("🔧 Preparing training set...\n");
trainTbl = mfccToTable(trainFeatures);

fprintf("🔧 Preparing test set...\n");
testTbl = mfccToTable(testFeatures);

X_train = vertcat(trainTbl.Features);
Y_train = categorical(trainTbl.Label);

X_test = vertcat(testTbl.Features);
Y_test = categorical(testTbl.Label);

%% === Train KNN Model ===
fprintf("🧠 Training KNN model (k=5)...\n");

knnModel = fitcknn(X_train, Y_train, ...
    'NumNeighbors', 5, ...
    'Standardize', true, ...
    'Distance', 'euclidean');

%% === Save Trained Model ===
modelPath = fullfile('models', 'knn_model.mat');
save(modelPath, 'knnModel');
fprintf("✅ KNN model saved at: %s\n", modelPath);

%% === Evaluate Model ===
fprintf("📈 Evaluating model on test set...\n");
Y_pred = predict(knnModel, X_test);

accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf("🎯 Test Accuracy: %.2f%%\n", accuracy * 100);

%% === Plot Confusion Matrix ===
figure;
confusionchart(Y_test, categorical(Y_pred));
title('KNN Confusion Matrix (MFCC Features)');
