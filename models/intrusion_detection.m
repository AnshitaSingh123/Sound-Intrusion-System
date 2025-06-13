% === SECTION 1: Select and Load Metadata File ===
[metaName, metaPath] = uigetfile('*.csv', 'Select the esc50.csv file');
metaFile = fullfile(metaPath, metaName);

% Check if file is valid
if ~isfile(metaFile)
    error('Metadata file not found: %s', metaFile);
end

% Load the metadata table
metadata = readtable(metaFile);
disp('Metadata loaded successfully.');

% === AUDIO FOLDER ===
audioFolder = 'audio_subset';

if ~isfolder(audioFolder)
    error('Audio folder not found.');
else
    disp('Audio folder loaded successfully.');
end

% === SELECT TARGET CLASSES ===
targetClasses = {'gun_shot', 'siren', 'engine', 'dog'};  % Modify as needed
filteredData = metadata(ismember(metadata.category, targetClasses), :);

% === INITIALIZE FEATURE AND LABEL ARRAYS ===
features = [];
labels = {};

disp(' Extracting MFCC features from audio files...');

for i = 1:height(filteredData)
    filename = fullfile(audioFolder, filteredData.filename{i});
    
    if ~isfile(filename)
        warning('Audio file not found: %s. Skipping...', filename);
        continue;
    end
    
    % Read audio
    [audioIn, fs] = audioread(filename);
    
    % Ensure mono channel
    if size(audioIn, 2) > 1
        audioIn = mean(audioIn, 2);
    end
    
    % Pad or trim to 5 seconds (ESC-50 clips are 5 seconds)
    targetLength = 5 * fs;
    if length(audioIn) < targetLength
        audioIn(end+1:targetLength) = 0;
    else
        audioIn = audioIn(1:targetLength);
    end
    
    % Extract MFCCs (13 coefficients), average over frames
    coeffs = mfcc(audioIn, fs, 'NumCoeffs', 13);
    meanMFCC = mean(coeffs, 1);
    
    % Store features and labels
    features = [features; meanMFCC];
    labels{end+1, 1} = filteredData.category{i};
end

labels = categorical(labels);
disp(' MFCC feature extraction complete.');

% === SPLIT DATA INTO TRAIN AND TEST SETS ===
cv = cvpartition(labels, 'HoldOut', 0.3);
XTrain = features(training(cv), :);
YTrain = labels(training(cv));
XTest = features(test(cv), :);
YTest = labels(test(cv));

% === TRAIN SVM CLASSIFIER ===
disp(' Training SVM classifier...');
model = fitcecoc(XTrain, YTrain);

% === PREDICT ON TEST SET ===
YPred = predict(model, XTest);

% === EVALUATE ACCURACY ===
accuracy = sum(YPred == YTest) / numel(YTest);
disp([' Test Accuracy: ', num2str(accuracy * 100, '%.2f'), '%']);

% === PLOT CONFUSION MATRIX ===
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix - ESC-50 Sound Intrusion Detection');

