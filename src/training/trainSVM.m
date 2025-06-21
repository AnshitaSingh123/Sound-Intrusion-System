% File: src/training/trainSVM.m
% Purpose: Train and evaluate an SVM using MFCC features

clc; clear;

%% === Load Pre-Extracted MFCC Features ===
fprintf("📂 Loading pre-extracted MFCC features...\n");
% Load the training and testing features from the specified .mat file
if isfile('src/feature_extraction/features_mfcc_spec.mat')
    fprintf("✅ Features file found. Proceeding to load...\n");
else
    error("❌ Features file not found. Please check the path.");
end
load('src/feature_extraction/features_mfcc_spec.mat', ...
    'trainFeatures', 'testFeatures');

%% === Convert struct array to table with averaged MFCCs ===
% This converts each MFCC matrix (13×T) into a 1×13 vector (mean over time)
mfccToTable = @(featureStruct) ...
    struct2table(arrayfun(@(x) struct( ...
        'Features', mean(x.mfcc, 2)', ...  % average MFCCs over time
        'Label', x.label), ...
        featureStruct));

fprintf("🔧 Preparing training set...\n");
trainTbl = mfccToTable(trainFeatures);

disp("🧪 Preview of trainTbl:");
disp(head(trainTbl));   % Shows Features and Label columns

% Convert feature vectors from table to matrix
X_train = vertcat(trainTbl.Features);   % Each row = 1×13 MFCC mean vector
Y_train = categorical(trainTbl.Label);  % Labels

fprintf("🔧 Preparing test set...\n");
testTbl = mfccToTable(testFeatures);

disp("🧪 Preview of testTbl:");
disp(head(testTbl));   % Optional: verify structure

X_test = vertcat(testTbl.Features);
Y_test = categorical(testTbl.Label);


%% === Train SVM Model ===
fprintf("🧠 Training SVM model (RBF kernel)...\n");

template = templateSVM('KernelFunction', 'rbf', 'Standardize', true);
svmModel = fitcecoc(X_train, Y_train, ...
    'Learners', template, ...
    'Coding', 'onevsall', ...
    'ClassNames', categories(Y_train));

%% === Save Trained Model ===
modelPath = fullfile('models', 'svm_model.mat');
save(modelPath, 'svmModel');
fprintf("✅ SVM model saved at: %s\n", modelPath);


%% === Evaluate Model ===
fprintf("📈 Evaluating model on test set...\n");
Y_pred = predict(svmModel, X_test);

accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf("🎯 Accuracy: %.2f%%\n", accuracy * 100);

%
%% === Plot Confusion Matrix ===
figure;
confusionchart(Y_test, categorical(Y_pred));
title('SVM Confusion Matrix (MFCC Features)');


