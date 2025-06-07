%% 🔧 Random Forest Training using MFCC features

clear; clc;

fprintf("📦 Loading pre-extracted MFCC features...\n");

% Load features
featuresPath = 'src/feature_extraction/features_mfcc_spec.mat';
if exist(featuresPath, 'file')
    load(featuresPath, 'trainFeatures', 'testFeatures');
else
    error("❌ Feature file not found: %s", featuresPath);
end

% Convert struct array to table with mean MFCC and label
mfccToTable = @(featureStruct) ...
    struct2table(arrayfun(@(x) struct( ...
        'Features', mean(x.mfcc, 2)', ...  % 1×13 vector from 13×N MFCC matrix
        'Label', x.label), ...
        featureStruct));

%% 🔧 Prepare training and test sets
fprintf("🔧 Preparing training & test sets...\n");

trainTbl = mfccToTable(trainFeatures);
testTbl  = mfccToTable(testFeatures);

X_train = vertcat(trainTbl.Features);  % n×13
Y_train = categorical(trainTbl.Label);

X_test = vertcat(testTbl.Features);
Y_test = categorical(testTbl.Label);

%% 🌲 Train Random Forest
fprintf("🚀 Training Random Forest (TreeBagger)...\n");
numTrees = 100;
rfModel = TreeBagger(numTrees, X_train, Y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'On', ...
    'OOBPredictorImportance', 'on');

fprintf("✅ Training completed!\n");

%% 🔍 Evaluate on test data
fprintf("📈 Predicting on test set...\n");
Y_pred = predict(rfModel, X_test);
Y_pred = categorical(Y_pred);  % Convert from cell to categorical

accuracy = mean(Y_pred == Y_test);
fprintf("🎯 Test Accuracy: %.2f%%\n", accuracy * 100);

% Confusion matrix
figure;
confusionchart(Y_test, Y_pred);
title('Random Forest Confusion Matrix (MFCC Features)');

%% 💾 Save model
modelPath = 'models/rfModel_MFCC.mat';
save(modelPath, 'rfModel');
fprintf("💾 Model saved to %s\n", modelPath);
